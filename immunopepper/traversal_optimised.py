from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Union, Optional
import numpy as np
import logging
import itertools
import timeit

import spladder.classes.gene
from immunopepper.dna_to_peptide import dna_to_peptide
from immunopepper.namedtuples import GeneTable, Coord, Flag, Peptide, OutputPeptide, OutputMetadata
from immunopepper.translate import complementary_seq, get_peptide_result
from immunopepper.filter import is_intron_in_junction_list, junctions_annotated
from immunopepper.mutations import get_mut_comb, get_mutated_sequence, mutation_to_seg_expression
from immunopepper.io_ import save_fg_peptide_set, namedtuple_to_str, save_kmer_matrix
from immunopepper.utils import replace_I_with_L, get_segment_expr_kmer
from immunopepper.preprocess import search_edge_metadata_segmentgraph, precompute_gene_junction_expressions


# A defaultdict to hold all valid segment paths derived from real transcripts
class SegmentPathIndex:
    def __init__(self):
        self.suffix_index: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        self.paths: List[List[int]] = [] # list of full transcript paths with seg_ids

    def insert(self, path: List[int]):
        """
        Insert full segment path and generate all contiguous subpaths (prefixes, infixes, suffixes)
        into the suffix_index.
        """
        self.paths.append(path)
        n = len(path)

        for i in range(n):
            for j in range(i + 1, n):
                subpath = tuple(path[i:j])
                next_seg = path[j]
                self.suffix_index[subpath].add(next_seg) # add next_seg to the set with the correct key (create if not yet existing)

    def children(self, partial_path: List[int]) -> List[int]:
        """
        Return valid next segments that can follow a given subpath (fast lookup).
        """
        return sorted(self.suffix_index.get(tuple(partial_path), [])) # return [] if subpath not in the index

    def get_all_paths(self) -> List[List[int]]:
        return self.paths

    def __str__(self):
        return '\n'.join(f"Path {i+1}: {path}" for i, path in enumerate(self.paths))

#TODO: potentially limit path length to 27 segments, as no kmer can span more than that
def build_segment_index(gene: spladder.classes.gene, exon_to_segments: dict) -> SegmentPathIndex:
    """
    Build a SegmentPathIndex of valid segment paths by traversing the splicegraph.
    Segment paths are derived from exon connectivity and segment-exon matches.
    If the strand is '-', the path is reversed at the end.

    Args:
        gene: An object containing the strand, splicegraph, and segmentgraph of a gene.
        exon_to_segments: A dict mapping exons and segment ids

    Returns:
        SegmentPathIndex: An index containing all valid segment paths and subpaths derived from the splicegraph.
    """
    index = SegmentPathIndex()
    splice_edges = gene.splicegraph.edges    # 2D boolean adjacency matrix (exons x exons)
    exon_coords = gene.splicegraph.vertices.T  # (N, 2) → [start, end] per exon
    terminals = gene.splicegraph.terminals # get terminal exons (based on the genomic coordinates, not translation direction)
    start_exons = list(np.where(terminals[0])[0]) # list for iteration
    end_exons = set(np.where(terminals[1])[0]) # set for fast lookup

    def is_forward(prev_exon: int, next_exon: int) -> bool:
        """
        Return True if moving forward along the splicegraph - going downstream in genomic coordinates
        """
        prev_start, prev_end = exon_coords[prev_exon]
        next_start, next_end = exon_coords[next_exon]
        return next_start > prev_start

    def dfs(exon_id: int, exon_path: List[int]):
        """
        Explore all valid exon paths starting from exon_id.
        """
        exon_path.append(exon_id)

        # If the current exon is an end terminal, convert the exon path to a segment path
        if exon_id in end_exons:
            segment_path = []
            for e_id in exon_path:
                segment_path.extend(exon_to_segments[e_id])

            # If on negative strand, reverse segment order
            if gene.strand == '-':
                segment_path = segment_path[::-1]
            index.insert(segment_path) # Insert the complete segment path into the index
            return

        # iterate over all possible next exons
        for next_exon in np.where(splice_edges[exon_id])[0]:
            if next_exon in exon_path: # if this exon already in the path 
                continue  # avoid cycles #TODO: ask if that ok! duplication events?
            if not is_forward(exon_id, next_exon): # check if the next exon downstream of the current one
                continue  # skip backward jumps
            # Recursively call dfs with a copy of current path extended by next_exon
            dfs(next_exon, exon_path[:])

    for start_exon in start_exons:
        dfs(start_exon, [])

    return index

def build_initial_kmers(cds_starts: List[int],
                        k: int,
                        segment_coords: np.ndarray,
                        strand: str,
                        index: SegmentPathIndex) -> List[List[Tuple[int, int, int]]]:
    """
    Build all valid k-mer paths (e.g., 27-mers) starting from CDS start positions.

    Args:
        cds_starts: List of genomic CDS start coordinates (0-based).
        k: k-mer length (number of nucleotides).
        segment_coords: 2xN array of genomic start and end positions per segment (gene.segmentgraph.segments).
        strand: '+' or '-' indicating gene orientation.
        index: A SegmentPathIndex used to validate segment paths.

    Returns:
        List of k-mer paths. Each path is a list of (segment_id, genomic_start, genomic_end) tuples.
        genomic_end is exclusive. 
    """
    results = []
    seen = set() # to avoid generating redundant initial kmers

    # Build a lookup for fast start coordinate → (segment_id, offset) resolution
    seg_coords = segment_coords.T  # Shape (N, 2), each row: (start, end)
    segment_lens = {i: abs(end - start) for i, (start, end) in enumerate(seg_coords)}

    # Helper: map genomic CDS start to segment ID and offset within the 1st segment
    def find_segment_and_offset(genomic_pos: int) -> Tuple[int, int]:
        # iterate over the segments
        for seg_id, (start, end) in enumerate(seg_coords):
            if strand == '+':
                # find in which segment cds starts
                if start <= genomic_pos < end:
                    # return segment id and how deep in the segment cds starts
                    return seg_id, genomic_pos - start
            else:  # negative strand
                if end > genomic_pos >= start:
                    return seg_id, end - genomic_pos    # no -1 here, already adjusted gtf coordinates in preprocess.preprocess_ann() #TODO: double-check
        return None, None  # Not found

    def dfs(current_id: int,
            current_offset: int,
            remaining: int,
            path: List[Tuple[int, int, int]],
            seg_path: List[int]):
        """Recursive DFS to collect valid k-mer paths along valid splice paths."""
        seg_len = segment_lens[current_id]
        if current_offset >= seg_len:
            return
        take = min(seg_len - current_offset, remaining)

        # Get genomic coordinates
        seg_start, seg_end = seg_coords[current_id]
        if strand == '+':
            cds_start = seg_start + current_offset
            cds_end = cds_start + take
        else:
            cds_start = seg_end - current_offset
            cds_end = cds_start - take

            # Make sure coordinates are in increasing order for output
            cds_start, cds_end = cds_end, cds_start

        new_path = path + [(current_id, cds_start, cds_end)]
        remaining -= take

        if remaining == 0:
            path_tuple = tuple(new_path)
            if path_tuple not in seen:
                results.append(new_path)
                seen.add(path_tuple)
            return

        # Propagate to all valid children in the index
        next_seg_ids = index.children(seg_path)
        for next_id in next_seg_ids:
            next_seg_path = seg_path + [next_id]
            dfs(next_id, 0, remaining, new_path, next_seg_path)

    # Start DFS from each CDS start position
    for cds_start in cds_starts:
        seg_id, offset = find_segment_and_offset(cds_start)
        if seg_id is not None:
            dfs(seg_id, offset, k, [], [seg_id])

    return results

def propagate_kmer(kmer: List[Tuple[int, int, int]],
                   segment_coords: np.ndarray,
                   strand: str,
                   index: SegmentPathIndex,
                   step: int = 3) -> List[List[Tuple[int, int, int]]]:
    """
    Propagate a k-mer (always 27nt) forward by step (def. 3 nt), respecting strand and valid segment paths.
    
    If step nt can't be added from the current segment, extend recursively through multiple
    valid child segments using index paths. If no child segments are available, the output will be an empty list.

    Args:
        kmer: Current k-mer as a list of (segment_id, start, end)
        segment_coords: 2 x N array with genomic coordinates of segments (start, end). gene.segmentgraph.segments
        strand: '+' or '-' indicating direction
        index: A SegmentPathIndex with valid segment continuations
        step: number of nucleotides by which to propagate a kmer in each step (default 3)

    Returns:
        A list of new propagated k-mers (as lists of (segment_id, start, end))
    """
    new_paths = []

    # Extract the segment path (list of segment IDs order)
    seg_path = [seg_id for seg_id, _, _ in kmer]

    # ----------- Step 1: Trim 3 nt (step) from the front -----------
    head_seg_id, head_start, head_end = kmer[0]
    head_len = head_end - head_start

    if head_len > step:
        # Just advance start of first segment
        if strand == '+':
            new_head = (head_seg_id, head_start + step, head_end)
        else: 
        # '-' strand, segment tuples look like: (3, 4900, 4980) so we want to be subtracting 3 (step) from the last el.
            new_head = (head_seg_id, head_start, head_end - step)
        trimmed_kmer = [new_head] + kmer[1:] # update the 1st segment
        new_seg_path = seg_path
    else:
        # Remove the first segment completely and subtract remaining from next
        trimmed_kmer = kmer[1:]
        new_seg_path = seg_path[1:] # remove 1st segment ID from the path
        if not trimmed_kmer:
            return []  # Nothing left after trimming

        # Adjust the new head segment by 3 (step) - head_len nt
        next_seg_id, next_start, next_end = trimmed_kmer[0]
        advance = step - head_len # how much is left after subtracting from the 1st segment
        if strand == '+':
            new_head = (next_seg_id, next_start + advance, next_end)
        else:
            new_head = (next_seg_id, next_start, next_end - advance)

        trimmed_kmer[0] = new_head
        
    # ----------- Step 2: Extend 3 nt (step) at the back -----------
    tail_seg_id, tail_start, tail_end = trimmed_kmer[-1]
    seg_start, seg_end = segment_coords[:, tail_seg_id]

    if strand == '+':
        seg_limit = seg_end
        remaining = seg_limit - tail_end # how much can we take from the current segment
        if remaining >= step: # enough space to propagate in the current segment
            new_tail = (tail_seg_id, tail_start, tail_end + step)
            new_paths.append(trimmed_kmer[:-1] + [new_tail])
        else:
            to_fill = step - remaining # how much will be taken from the next segment(s)
            base_path = trimmed_kmer
            if remaining > 0:
                tail_piece = (tail_seg_id, tail_start, tail_end + remaining)
                base_path = trimmed_kmer[:-1] + [tail_piece]

            def extend_forward(path, seg_ids, remaining_nt):
                children = index.children(seg_ids) # get a list of all the segments directly after the current kmer path #TODO: test
                for child_id in children: # propagate into all possible next segments
                    child_start, child_end = segment_coords[:, child_id]
                    child_len = child_end - child_start
                    take = min(child_len, remaining_nt) # take as much as possible from the next segment
                    next_piece = (child_id, child_start, child_start + take)
                    new_path = path + [next_piece] # list of kmers
                    new_seg_ids = seg_ids + [child_id] # list of seg ids
                    if take == remaining_nt: # next segment had enough nt, propagation done
                        new_paths.append(new_path)
                    elif child_len >= 1: # not done yet, extend in the next-next segment
                        extend_forward(new_path, new_seg_ids, remaining_nt - take)

            extend_forward(base_path, new_seg_path, to_fill)

    else:  
        # '-' strand, segment tuples look like: (3, 4900, 4980) so we want to be subtracting 3 (step) from the 2nd el.
        seg_limit = seg_start
        remaining = tail_start - seg_limit
        if remaining >= step: # enough space to propagate in the current segment
            new_tail = (tail_seg_id, tail_start - step, tail_end)
            new_paths.append(trimmed_kmer[:-1] + [new_tail])
        else: # not enough space in the current segment, need to go to the next one
            to_fill = step - remaining # how much will be taken from the next segment(s)
            base_path = trimmed_kmer
            if remaining > 0:
                tail_piece = (tail_seg_id, tail_start - remaining, tail_end)
                base_path = trimmed_kmer[:-1] + [tail_piece]

            def extend_reverse(path, seg_ids, remaining_nt):
                children = index.children(seg_ids) # get a list of all the segments directly after the current one #TODO: test
                for child_id in children: # propagate into all possible next segments
                    child_start, child_end = segment_coords[:, child_id]
                    child_len = child_end - child_start # ok, coordinates in the ascending order in the matrix
                    take = min(child_len, remaining_nt) # take as much as possible from the next segment
                    next_piece = (child_id, child_end - take, child_end) 
                    new_path = path + [next_piece] # list of kmer tuples
                    new_seg_ids = seg_ids + [child_id] # list of seg_ids
                    if take == remaining_nt: # next segment had enough nt, propagation done
                        new_paths.append(new_path)
                    elif child_len >= 1: # not done yet, extend in the next-next segment
                        extend_reverse(new_path, new_seg_ids, remaining_nt - take)

            extend_reverse(base_path, new_seg_path, to_fill)

    return new_paths

def get_exon_to_segments_dict(seg_match: np.ndarray) -> dict:
    return {exon_id: list(np.where(seg_match[exon_id])[0]) for exon_id in range(seg_match.shape[0])}

def get_segments_to_exons_dict(exon_to_segments: dict) -> dict:
    segment_to_exons = defaultdict(set)
    for exon, segs in exon_to_segments.items():
        for s in segs:
            segment_to_exons[s].add(exon)
    return segment_to_exons

def check_junction_annotation(kmer, gene, gene_annot_jx, junction_cache=None):
    """Check if any junction (intron-separated segments) in the k-mer path is annotated.
    gene_annot_jx is a set of junction strings in the format 'end_seg1:start_seg2'."""
    # Sort k-mer by segment ID to get genomic order
    sorted_kmer = sorted(kmer)
    
    for i in range(len(sorted_kmer) - 1):
        seg_id1, seg_id2 = sorted_kmer[i][0], sorted_kmer[i + 1][0]
        
        # Use junction cache to check if junction exists
        if junction_cache is not None:
            junction_key = (seg_id1, seg_id2)
            if junction_key in junction_cache:
                # Junction exists, get coordinates in genomic order
                seg_coord_list = gene.segmentgraph.segments
                end_seg1 = seg_coord_list[1, seg_id1]      # End of first segment
                start_seg2 = seg_coord_list[0, seg_id2]    # Start of second segment
                
                junction_str = f"{end_seg1}:{start_seg2}"
                if junction_str in gene_annot_jx:
                    return True
        else:
            # Fallback to direct matrix check for valid junctions
            if gene.segmentgraph.seg_edges[seg_id1, seg_id2]:
                seg_coord_list = gene.segmentgraph.segments
                end_seg1 = seg_coord_list[1, seg_id1]
                start_seg2 = seg_coord_list[0, seg_id2]
                
                junction_str = f"{end_seg1}:{start_seg2}"
                if junction_str in gene_annot_jx:
                    return True
    return False

def prepare_output_kmers(gene, idx, countinfo, seg_counts, edge_idxs, edge_counts,
                                     output_kmers, gene_annot_jx,
                                     graph_output_samples_ids,
                                     graph_samples, filepointer, out_dir, verbose=False):
    """
    Prepare output data for kmers and junctions for the entire gene, and write them to disk.
    """
    kmer_matrix_edge = []
    kmer_matrix_segm = []

    # Pre-compute junction expressions for this gene
    junction_cache = precompute_gene_junction_expressions(gene, edge_idxs, edge_counts)

    # Convert set to sorted list with custom key
    sorted_output_kmers = sorted(output_kmers, key=lambda x: [(seg_id, start, end) for seg_id, start, end in x[0]])

    # iterate over all the kmers for the gene
    # output_kmers is a set of tuples (kmer_coord, kmer_peptide, rf_annot, is_isolated)
    for kmer, kmer_peptide, rf_annot, is_isolated in sorted_output_kmers:
        k = len(kmer_peptide) #TODO: will always be the same length right or not? can stop shorten it?

        # get segment expression per segment per sample
        _, pos_expr_segm = get_segment_expr_kmer(gene, kmer, countinfo, seg_counts)
        # calculate length-weighted expression per sample, order of segments is irrelevant here
        sublist_seg = np.round(np.atleast_2d(pos_expr_segm[:, 0]).dot(pos_expr_segm[:, 1:]) / (k * 3), 2)
        sublist_seg = sublist_seg[0].tolist()

        # junction (edge) expression
        if (countinfo is not None) and not is_isolated:  # if there is countinfo and the kmer is not isolated (i.e., crosses junctions)
            _, edges_expr = search_edge_metadata_segmentgraph(gene, kmer, edge_idxs, edge_counts, junction_cache=junction_cache)

            # Get min expression value across all junctions for each sample
            sublist_jun = np.nanmin(edges_expr, axis=0)  # always apply. The min has no effect if one junction only
            if graph_output_samples_ids is not None:
                sublist_jun = sublist_jun[graph_output_samples_ids] # filter by sample
            sublist_jun = sublist_jun.tolist()
        else: # isolated kmer, no junctions
            sublist_jun = []

        # Flags: check whether junctions (genomic order!) in the kmer are annotated
        junction_annotated = False if is_isolated else check_junction_annotation(kmer, gene, gene_annot_jx, junction_cache=junction_cache)

        # create output data
        row_metadata = [kmer_peptide, ':'.join([f'{start}:{end}' for seg_id, start, end in kmer]),
                        not is_isolated, junction_annotated, rf_annot]
        if not is_isolated:  # if the kmer crosses junctions, save it in the edge matrix
            kmer_matrix_edge.append(row_metadata + sublist_jun)
        else:
            kmer_matrix_segm.append(row_metadata + sublist_seg)

        # save output data per batch
        if len(kmer_matrix_segm) > 5000:
            print(f'storing batch of {len(kmer_matrix_segm)} kmer_matrix_segm')
            time = timeit.default_timer()
            save_kmer_matrix(None, kmer_matrix_segm, graph_samples, filepointer, out_dir, verbose=False, gene_name=gene.name)
            print(f'done - took {timeit.default_timer() - time} seconds')
            kmer_matrix_segm.clear()
        if len(kmer_matrix_edge) > 5000:
            print(f'storing batch of {len(kmer_matrix_edge)} kmer_matrix_edge')
            time = timeit.default_timer()
            save_kmer_matrix(kmer_matrix_edge, None, graph_samples, filepointer, out_dir, verbose, gene_name=gene.name)
            print(f'done - took {timeit.default_timer() - time} seconds')
            kmer_matrix_edge.clear()

    save_kmer_matrix(kmer_matrix_edge, kmer_matrix_segm, graph_samples, filepointer, out_dir, verbose=False, gene_name=gene.name)

def get_and_write_kmer(
        gene: spladder.classes.gene.Gene,
        index: SegmentPathIndex,
        cds_starts: List[int],
        ref_mut_seq: str,
        segment_to_exons: dict,
        gene_annot_jx: set,
        mutation: object,
        kmer_length: int,
        idx: object = None,
        countinfo: object = None,
        edge_idxs: object = None,
        edge_counts: object = None,
        seg_counts: object = None,
        kmer_database: set = None,
        filepointer: object = None,
        graph_output_samples_ids: object = None,
        graph_samples: object = None,
        out_dir: str = None,
        verbose_save: bool = False,
    ) -> None:

    # k-mer path is a tuple of (segment_id, start, end)
    unique_kmers: Set[Tuple[Tuple[int, int, int], ...]] = set() # set to store seen kmer-paths
    output_kmers = set() # set to store final k-mers (kmer_path, kmer_peptide, rf_annot, is_isolated)
    queue: deque = deque() # Queue for k-mers to be propagated

    # Initialize 27-mers from CDS start positions
    init_paths = build_initial_kmers(cds_starts, kmer_length, gene.segmentgraph.segments, gene.strand, index)

    # iterate over initial kmers, get sequence, translate and check for STOP codons
    for path in init_paths:
        path_tuple = tuple(path)
        
        if path_tuple not in unique_kmers: # if this kmer is yet unseen (all mutated sequences have the same kmer coords, so will be repeated)
            unique_kmers.add(path_tuple) 
            has_any_valid_variant = False
            mut_seq_comb = get_mut_comb(path, mutation.somatic_dict) # get all possible comb. of somatic mutation positions

            # iterate over all the somatic mutation combinations and apply them to the reference sequence
            for variant_comb in mut_seq_comb:
                peptide, flag = get_peptide_result(path, gene.strand, variant_comb, mutation.somatic_dict, ref_mut_seq, gene.start, segment_to_exons)
                
                if not flag.has_stop:
                    has_any_valid_variant = True

                    # Remove peptides from a database on the fly
                    check_database = ((not kmer_database) or (replace_I_with_L(peptide.mut[0]) not in kmer_database))
                    if check_database: # add kmer_coord, kmer_peptide, rf_annot, is_isolated to the output_kmers set
                        output_kmers.add((path_tuple, peptide.mut[0], None, flag.is_isolated))  #TODO: None is read_frame_annotated, can be added later if needed

            # if at least one variant has no STOP, add to queue to propagate
            if has_any_valid_variant:
                queue.append(path)

    # Propagate k-mers (active paths) through the segment graph
    while queue: # While there are k-mers to propagate
        current_path = queue.popleft() # Remove and return a k-mer from the left side
        
        # Try to advance by step: 3 nt (--> 1 aa) 
        # new_paths is a list of kmers which is a lists of tuples (segment_id, start, end)
        new_paths = propagate_kmer(current_path, gene.segmentgraph.segments, gene.strand, index, step=3)

        # iterate over all possible next kmers
        for new_path in new_paths:
            path_tuple = tuple(new_path)

            # this will be true for alternative starts, which all lead to the same segment
            # this segment needs to be propagated only once, so we do not append it to queue again
            if path_tuple not in unique_kmers:
                unique_kmers.add(path_tuple)

                # for each next kmer, get sequences with all possible comb. of somatic mutations applied
                mut_seq_comb = get_mut_comb(new_path, mutation.somatic_dict)
                should_propagate = False # track if at least one of the mutated sequences has no STOP codon

                # iterate over all the somatic mutation combinations
                for variant_comb in mut_seq_comb:
                    peptide, flag = get_peptide_result(new_path, gene.strand, variant_comb, mutation.somatic_dict, ref_mut_seq, gene.start, segment_to_exons)

                    if not flag.has_stop:  # if no STOP codon, propagate further
                        should_propagate = True
                    
                        # Remove peptides from a database on the fly
                        check_database = ((not kmer_database) or (replace_I_with_L(peptide.mut[0]) not in kmer_database))
                        if check_database: # add kmer_coord, kmer_peptide, rf_annot to the output_kmers set
                            output_kmers.add((path_tuple, peptide.mut[0], None, flag.is_isolated))  #TODO: None is read_frame_annotated, can be added later if needed

                if should_propagate: # if at least one of the mutated sequences has no STOP codon
                    queue.append(new_path)  # no stop codon → continue propagating
    
    # Process and save all the kmers for the gene
    prepare_output_kmers(gene, idx, countinfo, seg_counts, edge_idxs, edge_counts,
                            output_kmers, gene_annot_jx,
                            graph_output_samples_ids,
                            graph_samples, filepointer, out_dir, verbose=verbose_save)
    return

def get_and_write_peptide(
        gene: spladder.classes.gene.Gene,
        index: SegmentPathIndex,
        cds_starts: List[int],
        ref_mut_seq: str,
        segment_to_exons: dict,
        mutation: object,
        som_exp_dict: dict,
        peptide_set: set,
        pep_length: int = 1000,
        pep_step: int = 30, # step size in amino acids, default 30 aa
        junction_list: set = None,
        filepointer: object = None,
        force_ref_peptides: bool = False,
        out_dir: str = None,
        fasta_save: bool = False,
        len_pep_save: int = 5000
    ) -> None:

    pep_step *= 3  # multiply by 3 to get nucleotide step size and to preserve RF

    unique_kmers: Set[Tuple[Tuple[int, int, int], ...]] = set() # store seen k-mers as tuples of (segment_id, start, end)
    queue: deque = deque() # Queue for k-mers to be propagated

    # Initialize 27-mers from CDS start positions
    init_paths = build_initial_kmers(cds_starts, pep_length, gene.segmentgraph.segments, gene.strand, index)

    # iterate over initial kmers, get sequence, translate and check for STOP codons
    for path in init_paths:
        path_tuple = tuple(path)
        
        if path_tuple not in unique_kmers: # if this kmer is yet unseen (all mutated sequences have the same kmer coords, so will be repeated)
            unique_kmers.add(path_tuple) 
            has_any_valid_variant = False
            mut_seq_comb = get_mut_comb(path, mutation.somatic_dict) # get all possible comb. of somatic mutation positions

            # iterate over all the somatic mutation combinations and apply them to the reference sequence
            variant_id = 0
            for variant_comb in mut_seq_comb:
                peptide, flag = get_peptide_result(path, gene.strand, variant_comb, mutation.somatic_dict, ref_mut_seq, gene.start, segment_to_exons)

                # Process and save peptides
                if not peptide.mut[0] \
                            or ((mutation.mode != 'ref') and (peptide.mut[0] in peptide.ref) and (not force_ref_peptides)):
                        continue

                # Use kmer path to get an unique kmer ID
                kmer_coord_string = f'{path_tuple[0][1]}:' + '-'.join(f'{seg}' for seg, start, end in path_tuple)
                new_output_id = f"{gene.name}_{kmer_coord_string}_{variant_id}"
               
                # Check if the intron defined by vertex_ids is in the user provided list of junctions
                is_intron_in_junction_list_flag = is_intron_in_junction_list(gene.splicegraph, path_tuple, gene.strand, junction_list)

                # collect expression data for each mutation position
                if not (isinstance(variant_comb, float) and np.isnan(variant_comb)) and som_exp_dict is not None:  # which means mutations exist
                    seg_exp_variant_comb = [int(som_exp_dict[ipos]) for ipos in variant_comb]
                else:
                    seg_exp_variant_comb = np.nan  # if no mutation or no count file,  the segment expression is .

                # Add peptide metadata to output
                peptide_set.add(namedtuple_to_str(OutputMetadata(peptide=peptide.mut[0],
                                    output_id=new_output_id,
                                    read_frame=None,
                                    read_frame_annotated=None, #TODO: can be added later if needed
                                    gene_name=gene.name,
                                    gene_chr=gene.chr,
                                    gene_strand=gene.strand,
                                    mutation_mode=mutation.mode,
                                    has_stop_codon=int(flag.has_stop),
                                    is_in_junction_list=is_intron_in_junction_list_flag,
                                    is_isolated=int(flag.is_isolated),
                                    variant_comb=variant_comb,
                                    variant_seg_expr=seg_exp_variant_comb,
                                    modified_exons_coord=':'.join([f'{start}-{end}' for seg_id, start, end in path_tuple]),
                                    original_exons_coord=None,
                                    vertex_idx=[seg for seg, _, _ in path_tuple],
                                    kmer_type=None
                                    ), sep = '\t'))
                variant_id += 1

                if len(peptide_set) > len_pep_save: # Save peptide batch to disk when threshold is reached
                    save_fg_peptide_set(peptide_set, filepointer, out_dir, fasta_save, verbose=False, gene_name=gene.name)
                    peptide_set.clear()
                
                if not flag.has_stop:
                    has_any_valid_variant = True

            # if at least one variant has no STOP, add to queue to propagate
            if has_any_valid_variant:
                queue.append(path)

    # Propagate k-mers (active paths) through the segment graph
    while queue: # While there are k-mers to propagate
        current_path = queue.popleft() # Remove and return a k-mer from the left side
        
        # Try to advance by step: 3 nt (--> 1 aa)
        # new_paths is a list of kmers which is a lists of tuples (segment_id, start, end)
        new_paths = propagate_kmer(current_path, gene.segmentgraph.segments, gene.strand, index, pep_step)

        # iterate over all possible next kmers
        for new_path in new_paths:
            path_tuple = tuple(new_path)

            # this will be true for alternative starts, which all lead to the same segment
            # this segment needs to be propagated only once, so we do not append it to queue again
            if path_tuple not in unique_kmers:
                unique_kmers.add(path_tuple)

                # for each next kmer, get sequences with all possible comb. of somatic mutations applied
                mut_seq_comb = get_mut_comb(new_path, mutation.somatic_dict)
                should_propagate = False # track if at least one of the mutated sequences has no STOP codon

                # iterate over all the somatic mutation combinations
                variant_id = 0
                for variant_comb in mut_seq_comb:
                    peptide, flag = get_peptide_result(new_path, gene.strand, variant_comb, mutation.somatic_dict, ref_mut_seq, gene.start, segment_to_exons)

                    # Process and save peptides
                    if not peptide.mut[0] \
                                or ((mutation.mode != 'ref') and (peptide.mut[0] in peptide.ref) and (not force_ref_peptides)):
                            continue

                    # Use kmer path to get an unique kmer ID
                    kmer_coord_string = f'{path_tuple[0][1]}:' + '-'.join(f'{seg}' for seg, start, end in path_tuple)
                    new_output_id = f"{gene.name}_{kmer_coord_string}_{variant_id}"
                
                    # Check if the intron defined by vertex_ids is in the user provided list of junctions
                    is_intron_in_junction_list_flag = is_intron_in_junction_list(gene.splicegraph, path_tuple, gene.strand, junction_list)

                    # collect expression data for each mutation position
                    if not (isinstance(variant_comb, float) and np.isnan(variant_comb)) and som_exp_dict is not None:  # which means mutations exist
                        seg_exp_variant_comb = [int(som_exp_dict[ipos]) for ipos in variant_comb]
                    else:
                        seg_exp_variant_comb = np.nan  # if no mutation or no count file,  the segment expression is .

                    # Add peptide metadata to output
                    peptide_set.add(namedtuple_to_str(OutputMetadata(peptide=peptide.mut[0],
                                        output_id=new_output_id,
                                        read_frame=None,
                                        read_frame_annotated=None, #TODO: can be added later if needed
                                        gene_name=gene.name,
                                        gene_chr=gene.chr,
                                        gene_strand=gene.strand,
                                        mutation_mode=mutation.mode,
                                        has_stop_codon=int(flag.has_stop),
                                        is_in_junction_list=is_intron_in_junction_list_flag,
                                        is_isolated=int(flag.is_isolated),
                                        variant_comb=variant_comb,
                                        variant_seg_expr=seg_exp_variant_comb,
                                        modified_exons_coord=':'.join([f'{start}-{end}' for seg_id, start, end in path_tuple]),
                                        original_exons_coord=None,
                                        vertex_idx=[seg for seg, _, _ in path_tuple],
                                        kmer_type=None
                                        ), sep = '\t'))
                    variant_id += 1

                    if len(peptide_set) > len_pep_save: # Save peptide batch to disk when threshold is reached
                        save_fg_peptide_set(peptide_set, filepointer, out_dir, fasta_save, verbose=False, gene_name=gene.name)
                        peptide_set.clear()

                    if not flag.has_stop:  # if no STOP codon, propagate further
                        should_propagate = True

                if should_propagate: # if at least one of the mutated sequences has no STOP codon
                    queue.append(new_path)  # no stop codon → continue propagating

    # Save the last batch of peptides
    save_fg_peptide_set(peptide_set, filepointer, out_dir, fasta_save, verbose=False, gene_name=gene.name)
    return

def get_kmers_and_peptides(
        gene: spladder.classes.gene.Gene,
        mutation: object,
        table: object,
        ref_seq_file: str,
        chrm: str,
        peptide_set: set,
        kmer_length: int = 27,
        pep_length: int = 999,
        idx: object = None,
        countinfo: object = None,
        edge_idxs: object = None,
        edge_counts: object = None,
        seg_counts: object = None,
        mut_count_id: object = None,
        junction_list: set = None,
        kmer_database: set = None,
        filepointer: object = None,
        force_ref_peptides: bool = False,
        graph_output_samples_ids: object = None,
        graph_samples: object = None,
        out_dir: str = None,
        verbose_save: bool = False,
        fasta_save: bool = False
    ) -> None:
    """
    Traverse a splicing graph twice and generate:
        - k-mers (9 aa) potential neoepitopes
        - peptides (333 aa) for MS enzymataic digestion
    Apply somatic mutations, filter against reference databases, and write
    k-mer expression data and peptide sequences to disk.

    Parameters
    ----------
    gene : spladder.classes.gene.Gene
        Gene object containing splice graph, segment graph, and genomic coordinates
    mutation : object
        Mutation object with somatic_dict, germline_dict, and mode attributes
    table : object
        Gene table with CDS coordinates and transcript mappings
    ref_seq_file : str
        Path to reference genome FASTA file
    chrm : str
        Chromosome name for sequence extraction
    kmer_length : int
        Length of k-mers to generate (nucleotides)
    pep_length : int
        Maximum length of peptides to generate (nucleotides)
    peptide_set : set
        Set to accumulate peptide metadata for batch saving
    idx : object, optional
        Index information with gene and sample attributes
    countinfo : object, optional
        SplAdder count information for expression quantification
    edge_idxs : object, optional
        Edge indices for junction expression calculation
    edge_counts : object, optional
        Edge count data for junction expression
    seg_counts : object, optional
        Segment count data for expression quantification
    mut_count_id : object, optional
        Column indices for mutation sample in expression matrix
    junction_list : set, optional
        Set of junction coordinates to filter against
    kmer_database : set, optional
        Set of k-mer amino acid sequences to exclude (e.g., from UniProt)
    filepointer : object, optional
        File paths and column information for output files
    force_ref_peptides : bool, default False
        Whether to include mutated peptides identical to reference
    graph_output_samples_ids : object, optional
        Sample indices for expression output
    graph_samples : object, optional
        Sample names for expression matrix headers
    out_dir : str, optional
        Output directory for temporary and final files
    verbose_save : bool, default False
        Whether to print verbose output during saving
    fasta_save : bool, default False
        Whether to save peptides in FASTA format

    Returns
    -------
    None
        Function writes output files directly to disk
    """
    
    # -------------- from collect_vertex_pairs
    gene.from_sparse()

    # 1) apply germline mutations to the reference sequence
    # when germline mutation is applied, background_seq != ref_seq
    # otherwise, background_seq = ref_seq
    ref_mut_seq = get_mutated_sequence(fasta_file=ref_seq_file,
                                       chromosome=chrm,
                                       pos_start=gene.splicegraph.vertices.min(),
                                       pos_end=gene.splicegraph.vertices.max(),
                                       mutation_dict=mutation.germline_dict)

    # -------------- from get_and_write_peptide_and_kmer
    # check whether the junction (specific combination of vertices) also is annotated as a  junction of a protein coding transcript
    # 1) return set of all the junctions pairs of a gene {"exon1_end:exon2_start"} in genomic order appearing in any transcript given by .gtf file
    gene_annot_jx = junctions_annotated(gene, table.gene_to_ts, table.ts_to_cds)
    # 2) get a dictionary mapping somatic mutation positions to segment expression data
    som_exp_dict = mutation_to_seg_expression(gene, list(mutation.somatic_dict.keys()), countinfo, seg_counts, mut_count_id) # return a dictionary mapping exon ids to expression data
    
    # -------------- my code
    # Get a list of all annotated cds start coordinates for the gene
    cds_starts = list(set(table.gene_to_cds_begin[gene.name][transcript][0] for transcript in range(len(table.gene_to_cds_begin[gene.name])))) #"gene name", transcript ID
    
    # Map exons and segment IDs
    exon_to_segments = get_exon_to_segments_dict(gene.segmentgraph.seg_match)
    segment_to_exons = get_segments_to_exons_dict(exon_to_segments)

    # Build an index of valid continuations from each segment
    index = build_segment_index(gene, exon_to_segments)

    # Get kmers for the gene
    get_and_write_kmer(gene, index, cds_starts, ref_mut_seq, segment_to_exons, gene_annot_jx, 
        mutation, kmer_length, idx,countinfo, edge_idxs, edge_counts, seg_counts, kmer_database,
        filepointer,graph_output_samples_ids, graph_samples, out_dir, verbose_save)
    
    # Get peptides for the gene
    get_and_write_peptide(gene, index, cds_starts, ref_mut_seq, segment_to_exons, 
        mutation, som_exp_dict, peptide_set, pep_length=pep_length, pep_step=30,
        junction_list=junction_list, filepointer=filepointer,
        force_ref_peptides=force_ref_peptides, out_dir=out_dir,
        fasta_save=fasta_save, len_pep_save=5000)

    if not gene.splicegraph.edges is None:
        gene.to_sparse()
    return