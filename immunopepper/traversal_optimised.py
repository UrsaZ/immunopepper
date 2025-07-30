from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import logging
import itertools

from immunopepper.dna_to_peptide import dna_to_peptide
from immunopepper.namedtuples import GeneTable, Coord, Flag, Peptide
from immunopepper.translate import complementary_seq
from immunopepper.filter import is_intron_in_junction_list

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

def build_segment_index(gene) -> SegmentPathIndex:
    """
    Build a SegmentPathIndex of valid segment paths by traversing the splicegraph.
    Segment paths are derived from exon connectivity and segment-exon matches.
    If the strand is '-', the path is reversed at the end.

    Args:
        gene: An object containing the strand, splicegraph, and segmentgraph of a gene.

    Returns:
        SegmentPathIndex: An index containing all valid segment paths and subpaths derived from the splicegraph.
    """
    index = SegmentPathIndex()
    seg_match = gene.segmentgraph.seg_match  # 2D boolean matrix (exons x segments) 
    splice_edges = gene.splicegraph.edges    # 2D boolean adjacency matrix (exons x exons)
    exon_coords = gene.splicegraph.vertices.T  # (N, 2) → [start, end] per exon

    # Map exon → list of segment IDs
    exon_to_segments = {
        exon_id: list(np.where(seg_match[exon_id])[0])
        for exon_id in range(seg_match.shape[0])}

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
                   index: SegmentPathIndex) -> List[List[Tuple[int, int, int]]]:
    """
    Propagate a k-mer (always 27nt) forward by 3 nt, respecting strand and valid segment paths.
    
    If 3 nt can't be added from the current segment, extend recursively through multiple
    valid child segments using index paths. If no child segments are available, the output will be an empty list.

    Args:
        kmer: Current k-mer as a list of (segment_id, start, end)
        segment_coords: 2 x N array with genomic coordinates of segments (start, end). gene.segmentgraph.segments
        strand: '+' or '-' indicating direction
        index: A SegmentPathIndex with valid segment continuations

    Returns:
        A list of new propagated k-mers (as lists of (segment_id, start, end))
    """
    new_paths = []

    # Extract the segment path (list of segment IDs order)
    seg_path = [seg_id for seg_id, _, _ in kmer]

    # ----------- Step 1: Trim 3 nt from the front -----------
    head_seg_id, head_start, head_end = kmer[0]
    head_len = head_end - head_start

    if head_len > 3:
        # Just advance start of first segment
        if strand == '+':
            new_head = (head_seg_id, head_start + 3, head_end)
        else: 
        # '-' strand, segment tuples look like: (3, 4900, 4980) so we want to be subtracting 3 from the last el.
            new_head = (head_seg_id, head_start, head_end - 3)
        trimmed_kmer = [new_head] + kmer[1:] # update the 1st segment
        new_seg_path = seg_path
    else:
        # Remove the first segment completely and subtract remaining from next
        trimmed_kmer = kmer[1:]
        new_seg_path = seg_path[1:] # remove 1st segment ID from the path
        if not trimmed_kmer:
            return []  # Nothing left after trimming

        # Adjust the new head segment by 3 - head_len nt
        next_seg_id, next_start, next_end = trimmed_kmer[0]
        advance = 3 - head_len # how much is left after subtracting from the 1st segment
        if strand == '+':
            new_head = (next_seg_id, next_start + advance, next_end)
        else:
            new_head = (next_seg_id, next_start, next_end - advance)

        trimmed_kmer[0] = new_head
        
    # ----------- Step 2: Extend 3 nt at the back -----------
    tail_seg_id, tail_start, tail_end = trimmed_kmer[-1]
    seg_start, seg_end = segment_coords[:, tail_seg_id]

    if strand == '+':
        seg_limit = seg_end
        remaining = seg_limit - tail_end # how much can we take from the current segment
        if remaining >= 3: # enough space to propagate in the current segment
            new_tail = (tail_seg_id, tail_start, tail_end + 3)
            new_paths.append(trimmed_kmer[:-1] + [new_tail])
        else:
            to_fill = 3 - remaining # how much will be taken from the next segment(s)
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
        # '-' strand, segment tuples look like: (3, 4900, 4980) so we want to be subtracting 3 from the 2nd el.
        seg_limit = seg_start
        remaining = tail_start - seg_limit
        if remaining >= 3: # enough space to propagate in the current segment
            new_tail = (tail_seg_id, tail_start - 3, tail_end)
            new_paths.append(trimmed_kmer[:-1] + [new_tail])
        else: # not enough space in the current segment, need to go to the next one
            to_fill = 3 - remaining # how much will be taken from the next segment(s)
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

#TODO: replace mutations.get_mut_comb()
def kmer_to_mutations(kmer: List[Tuple[int, int, int]], mutation_pos: dict) -> List[Tuple]:
    """
    Returns all non-empty subsets of mutations that fall within the kmer.

    Args:
        kmer: list of (seg_id, start, stop) tuples.
        mutation_pos: dict of mutation positions (genomic coords as keys).

    Returns:
        List of tuples representing mutation combinations (excluding empty set).
    """
    mut_set = set()
    for seg_id, start, stop in kmer:
        for mutation in mutation_pos:
            if start <= mutation < stop:
                mut_set.add(mutation)
                
    def kmer_to_mutations(kmer: List[Tuple[int, int, int]], mutation_pos: dict) -> List[Tuple]:
    """
    Returns all non-empty subsets of mutations that fall within the kmer.

    Args:
        kmer: list of (seg_id, start, stop) tuples.
        mutation_pos: dict of mutation positions (genomic coords as keys).

    Returns:
        List of tuples representing mutation combinations (excluding empty set).
    """
    mut_set = set()
    for seg_id, start, stop in kmer:
        for mutation in mutation_pos:
            if start <= mutation < stop:
                mut_set.add(mutation)
                
    if not mut_set:
        return np.nan

    mut_list = sorted(mut_set)  # for reproducibility
    combs = [comb for i in range(1, len(mut_list) + 1)
             for comb in itertools.combinations(mut_list, i)]
    return combs

#TODO: replace utils.get_sub_mut_dna with this one
def get_sub_mut_dna(background_seq: str,
                    kmer: List[Tuple[int, int, int]],
                    variant_comb: List[int],
                    somatic_mutation_sub_dict: Dict[int, Dict[str, str]],
                    strand: str,
                    gene_start: int) -> str:
    """ Get the mutated dna sub-sequence according to mutation specified by the variant_comb.

    Parameters
    ----------
    background_seq: List(str). backgound sequence.
    kmer: List of (seg_id, start, end) genomic coordinate tuples.
    variant_comb: List(int). List of variant position. Like ['38', '43']
    somatic_mutation_sub_dict: Dict. variant position -> variant details.
    strand: gene strand

    Returnvariant_combs
    -------
    sub_dna: str. dna when applied somatic mutation (reverse for '-' strand).

    """
    def _get_variant_pos_offset(variant_pos, kmer, strand):
        """
        Convert variant's genomic position to a relative position in the kmer.
        """
        offset = 0 # position of the variant within the flattened DNA string in translational order, 0-based
        takes_effect = False # variant lies on the exon
        
        for seg_id, start, end in kmer:
            if variant_pos >= start and variant_pos < end: # variant inside current segment
                if strand == '+':
                    offset += variant_pos - start # add offset within current segment
                else:
                    offset += end - variant_pos - 1 # rel. position from the end, -1 beacuse end is exclusive
                takes_effect = True
                break
            else:
                # If mutation not in the current segment, add the full length of the segment to the offset 
                offset += end - start

        return offset if takes_effect else np.nan

    if strand == '+': # concatenate exon slices from background_seq
        sub_dna = ''.join([background_seq[start - gene_start:end - gene_start] for seg_id, start, end in kmer])
    else: # for '-': reverse slice per pair, no complement yet so that we can apply mutations
        sub_dna = ''.join([background_seq[start - gene_start:end - gene_start][::-1] for seg_id, start, end in kmer])
    if variant_comb is np.nan:  # no mutation exist
        return sub_dna

    # Apply mutations
    relative_variant_pos = [_get_variant_pos_offset(variant_ipos, kmer, strand) for variant_ipos in variant_comb]
    for i, variant_ipos in enumerate(variant_comb):
        # get ref and mutated base from the mutation dict
        mut_base = somatic_mutation_sub_dict[variant_ipos]['mut_base'] 
        ref_base = somatic_mutation_sub_dict[variant_ipos]['ref_base']
        pos = relative_variant_pos[i] # get relative position
        if not np.isnan(pos): # if mutation covered by a kmer
            sub_dna = sub_dna[:pos] + mut_base + sub_dna[pos+1:]
    return sub_dna

#TODO: replace translate.get_peptide_result with this
def get_peptide_result(kmer: List[Tuple[int, int, int]],
                       strand: str,
                       variant_comb: List[int],
                       somatic_mutation_sub_dict: Dict[int, Dict[str, str]],
                       ref_mut_seq: Dict[str, str],
                       gene_start: int,
                       all_read_frames: bool = False) -> Tuple["Peptide", "Flag"]:
    """
    Generate mutated and reference peptides from a kmer and variant combination.

    Parameters
    ----------
    kmer: List of (seg_id, start, stop) tuples representing genomic segments.
    strand: '+' or '-'.
    variant_comb: List of variant positions.
    somatic_mutation_sub_dict: Dict from variant pos to mutation details.
    ref_mut_seq: Dict with keys 'ref' and 'background' sequences.
    gene_start: Genomic start coordinate of the gene.
    all_read_frames: if false, only the first peptide until the stop codon is returned, 
    otherwise a list of all translated peptides (for each stop codon) is provided.

    Returns
    -------
    Tuple of Peptide and Flag objects.
    """

    # Choose correct background/reference sequences
    if somatic_mutation_sub_dict:
        ref_seq = ref_mut_seq['background']
    else:
        ref_seq = ref_mut_seq['ref']

    mut_seq = ref_mut_seq['background']

    # Get sub-DNA strings (mutated and reference)
    dna_str_mut = get_sub_mut_dna(mut_seq, kmer, variant_comb, somatic_mutation_sub_dict, strand, gene_start)
    dna_str_ref = get_sub_mut_dna(ref_seq, kmer, np.nan, somatic_mutation_sub_dict, strand, gene_start)

    # Generate a complement for '-' strand (reverse done inside get_sub_mut_dna)
    if strand == "-":
        dna_str_mut = complementary_seq(dna_str_mut)
        dna_str_ref = complementary_seq(dna_str_ref)

    # Translate DNA to peptide
    peptide_mut, mut_has_stop_codon = dna_to_peptide(dna_str_mut, all_read_frames)
    peptide_ref, ref_has_stop_codon = dna_to_peptide(dna_str_ref, all_read_frames)

    # if the stop codon appears before translating the second exon, mark 'single' #FIXME: can be more than one segment, but still only one exon!!!
    if len(kmer) < 2 or len(peptide_mut[0])*3 <= abs(kmer[0][2] - kmer[0][1]) + 1:
        is_isolated = True
    else:
        is_isolated = False
        
    # Wrap results #TODO: potentially output DNA seq as well
    peptide = Peptide(peptide_mut, peptide_ref)
    flag = Flag(mut_has_stop_codon, is_isolated)
    return peptide, flag

def get_kmers_and_translate(gene,
                             ref_mut_seq: str,
                             genetable: GeneTable,
                             k: int = 27,
                             stop_on_stop: bool = True) -> Set[Tuple[Tuple[int, int, int], ...]]:
    """
    Extract unique k-mers from a segment graph using propagation strategy.

    Parameters:
        gene: Gene object with .strand, .segmentgraph and .splicegraph
        ref_mut_seq: a dict with reference and mutated sequences for the gene
        genetable: NamedTuple with gene-transcript-cds mapping tables derived from .gtf file. 
                Has attributes ['gene_to_cds_begin', 'ts_to_cds', 'gene_to_ts']
        k: k-mer size (default 27)
        stop_on_stop: whether to stop propagating kmers with in-frame STOP codons

    Returns:
        Set of unique k-mers, each as a tuple of (segment_id, start, end)
    """
    #TODO: decide if gene.from_sparse() will be done here

    # Get a list of all annotated cds start coordinates for the gene
    cds_starts = list(set(genetable.gene_to_cds_begin[gene.name][transcript][0] for transcript in range(len(genetable.gene_to_cds_begin[gene.name])))) #"gene name", transcript ID

    # Build an index of valid segment paths from actual transcripts
    index = build_segment_index(gene)

    # Set to store final k-mers as tuples of (segment_id, start, end)
    unique_kmers: Set[Tuple[Tuple[int, int, int], ...]] = set()

    # Queue for k-mers to be propagated 
    # deque is a list-like container with fast appends and pops on either end
    queue: deque = deque()

    # Initialize 27-mers from CDS start positions
    init_paths = build_initial_kmers(cds_starts, k, gene.segmentgraph.segments, gene.strand, index)

    # iteravte over kmers, get sequence, translate and check for STOP codons
    for path in init_paths:
        path_tuple = tuple(path)
        # get sequences with all possible comb. of somatic mutations applied
        mut_seq_comb = kmer_to_mutations(path, sub_mutation.somatic_dict)
        for variant_comb in mut_seq_comb:
            peptide, flag = get_peptide_result(path, gene.strand, variant_comb, sub_mutation.somatic_dict, ref_mut_seq, gene.start)

            if stop_on_stop:
                if flag.has_stop:
                    #TODO: ask what to do with short initial kmers
                    continue
                # if no STOP in the initial kmer, save to results and add to queue to propagate
            if path_tuple not in unique_kmers: # if this kmer is yet unseen
                queue.append(path)
                unique_kmers.add(path_tuple) #TODO: save the seq as well

    # Propagate k-mers (active paths) through the segment graph
    # the graph is traversed in the direction of the translation, not transcript by transcript.
    while queue: # While there are k-mers to propagate
        current_path = queue.popleft() # Remove and return a k-mer from the left side
        
        # Try to advance by 3 nt (--> 1 aa)
        # new_paths is a list of kmers which is a lists of tuples (segment_id, start, end)
        new_paths = propagate_kmer(current_path, gene.segmentgraph.segments, gene.strand, index)

        # iterate over all possible next kmers
        for new_path in new_paths:
            path_tuple = tuple(new_path)

            # this will be true for alternative starts, which all lead to the same segment
            # this segment needs to be propagated only once, so we do not append it to queue again
            if path_tuple not in unique_kmers:
                unique_kmers.add(path_tuple) #TODO: save the seq as well

                # for each next kmer, get sequences with all possible comb. of somatic mutations applied
                mut_seq_comb = kmer_to_mutations(path, sub_mutation.somatic_dict)
                for variant_comb in mut_seq_comb:
                    peptide, flag = get_peptide_result(new_path, gene.strand, variant_comb, sub_mutation.somatic_dict, ref_mut_seq, gene.start)
                    
                    if stop_on_stop:
                        if flag.has_stop:
                            continue
                    if new_path not in queue:
                        queue.append(new_path)  # no stop codon → continue propagating

    return unique_kmers

def get_and_write_peptide_and_kmer(peptide_set: object = None, 
                                   gene: object = None, ref_mut_seq: object = None, idx: object = None,
                                   exon_som_dict: object = None, countinfo: object = None,
                                   edge_idxs: object = None, edge_counts: object = None, seg_counts: object = None,
                                   mutation: object = None, mut_count_id: object = None, table: object = None,
                                   junction_list: object = None, kmer_database: object = None,
                                   filepointer: object = None,
                                   force_ref_peptides: object = False, kmer: object = None,
                                   all_read_frames: object = None, graph_output_samples_ids: object = None,
                                   graph_samples: object = None, out_dir: object = None, verbose_save: object = None,
                                   fasta_save: object = None, k: int = 27) -> object:
    """
    Traverse the splice graph, get kmers, translate them to peptide sequences, 
    processes mutations, and write to disk.

    Parameters
    ----------
    peptide_set: set(OutputMetadata, OutputMetadata) with OutputMetadata namedtuple
    gene: Object, returned by SplAdder.
    ref_mut_seq: Str, reference sequnce of specific chromosome
    idx: Namedtuple Idx, has attribute idx.gene and idx.sample
    exon_som_dict: Dict. (exon_id) |-> (mutation_postion)
    countinfo: Namedtuple, contains SplAdder count information
    mutation: Namedtuple Mutation, store the mutation information of specific chromosome and sample.
        has the attribute ['mode', 'maf_dict', 'vcf_dict']
    table: Namedtuple GeneTable, store the gene-transcript-cds mapping tables derived
       from .gtf file. has attribute ['gene_to_cds_begin', 'ts_to_cds', 'gene_to_cds']
    size_factor: Scalar. To adjust the expression counts based on the external file `libsize.tsv`
    junction_list: List. Work as a filter to indicate some exon pair has certain
       ordinary intron which can be ignored further.
    kmer_database: Set. kmers to be removed on the fly from the kmer sample or matrix files
    filepointer: namedtuple, contains the columns and paths of each file of interest
    force_ref_peptides: bool, flag indicating whether to force output of
        mutated peptides which are the same as reference peptides
    kmer: list containing the length of the kmers requested
    out_dir: str, base direactory used for temporary files
    graph_samples: list, samples contained in the splicing graph object
    fasta_save: bool. whether to save a fasta file with the peptides
    """
    len_pep_save = 9999 # save at most this many peptides in the set before writing to file
    
    # check whether the junction (specific combination of vertices) also is annotated as a  junction of a protein coding transcript
    # 1) return set of all the junctions pairs of a gene {"exon1_end:exon2_start"}
    gene_annot_jx = junctions_annotated(gene, table.gene_to_ts, table.ts_to_cds) #TODO: get segment junctions
    # 2) get a dictionary mapping exon ids to expression data.
    som_exp_dict = exon_to_expression(gene, list(mutation.somatic_dict.keys()), countinfo, seg_counts, mut_count_id) # return a dictionary mapping exon ids to expression data
    
    #TODO: started updating from here on
    # Get a list of all annotated cds start coordinates for the gene
    cds_starts = list(set(table.gene_to_cds_begin[gene.name][transcript][0] for transcript in range(len(table.gene_to_cds_begin[gene.name])))) #"gene name", transcript ID

    # Build an index of valid segment paths from actual transcripts
    index = build_segment_index(gene)

    # Set to store final k-mers as tuples of (segment_id, start, end)
    unique_kmers: Set[Tuple[Tuple[int, int, int], ...]] = set()
    # Queue for k-mers to be propagated (list-like container with fast appends and pops on either end)
    queue: deque = deque()

    # Initialize 27-mers from CDS start positions
    init_paths = build_initial_kmers(cds_starts, k, gene.segmentgraph.segments, gene.strand, index)

    # iteravte over initial kmers, get sequence, translate and check for STOP codons
    for path in init_paths:
        path_tuple = tuple(path)
        # get sequences with all possible comb. of somatic mutations applied
        mut_seq_comb = kmer_to_mutations(path, sub_mutation.somatic_dict)
        for variant_comb in mut_seq_comb:
            peptide, flag = get_peptide_result(path, gene.strand, variant_comb, sub_mutation.somatic_dict, ref_mut_seq, gene.start)
#TODO: continue
            if stop_on_stop:
                if flag.has_stop:
                    #TODO: ask what to do with short initial kmers
                    continue
                # if no STOP in the initial kmer, save to results and add to queue to propagate
            if path_tuple not in unique_kmers: # if this kmer is yet unseen
                queue.append(path)
                unique_kmers.add(path_tuple) #TODO: save the seq as well

    # Propagate k-mers (active paths) through the segment graph
    # the graph is traversed in the direction of the translation, not transcript by transcript.
    while queue: # While there are k-mers to propagate
        current_path = queue.popleft() # Remove and return a k-mer from the left side
        
        # Try to advance by 3 nt (--> 1 aa)
        # new_paths is a list of kmers which is a lists of tuples (segment_id, start, end)
        new_paths = propagate_kmer(current_path, gene.segmentgraph.segments, gene.strand, index)

        # iterate over all possible next kmers
        for new_path in new_paths:
            path_tuple = tuple(new_path)

            # this will be true for alternative starts, which all lead to the same segment
            # this segment needs to be propagated only once, so we do not append it to queue again
            if path_tuple not in unique_kmers:
                unique_kmers.add(path_tuple) #TODO: save the seq as well

                # for each next kmer, get sequences with all possible comb. of somatic mutations applied
                mut_seq_comb = kmer_to_mutations(path, sub_mutation.somatic_dict)
                # iterate over all the somatic mutation combinations
                for variant_comb in mut_seq_comb:
                    peptide, flag = get_peptide_result(new_path, gene.strand, variant_comb, sub_mutation.somatic_dict, ref_mut_seq, gene.start)

                    if stop_on_stop:
                        if flag.has_stop:
                            continue
                    if new_path not in queue:
                        queue.append(new_path)  # no stop codon → continue propagating

                    # If cross junction peptide has a stop-codon in it, the frame
                    # will not be propagated because the read is truncated before it reaches the end of the exon.
                    # also in mutation mode, only output the case where ref is different from mutated
                    
                    # iterate over all the peptides (can be more, if more RFs) in the peptide object
                    for pep_idx in np.arange(len(peptide.mut)):
                        # Do not output peptide if:
                        # (1) peptide is empty peptide
                        # (2) In mutation mode the peptide is the same as the reference, unless the user forced redundancy
                        if not peptide.mut[pep_idx] \
                                or ((mutation.mode != 'ref') and (peptide.mut[pep_idx] in peptide.ref) and (not force_ref_peptides)):
                            continue

                        # collect flags
                        # generate a unique output id for the peptide
                        new_output_id = ':'.join([gene.name, '_'.join([str(v) for v in vertex_list]), str(variant_id), str(tran_start_pos), kmer_type])
                        # Check if the intron defined by vertex_ids is in the user provided list of junctions
                        is_intron_in_junction_list_flag = is_intron_in_junction_list(gene.splicegraph, vertex_list, gene.strand, junction_list) #FIXME: update for kmers

                        # collect expression data for each mutation position
                        if variant_comb is not np.nan and som_exp_dict is not None:  # which means mutations exist
                            seg_exp_variant_comb = [int(som_exp_dict[ipos]) for ipos in variant_comb]
                        else:
                            seg_exp_variant_comb = np.nan  # if no mutation or no count file,  the segment expression is .


                        ### Peptides
                        # Add peptide metadata to output
                        peptide_set.add(namedtuple_to_str(OutputMetadata(peptide=peptide.mut[pep_idx],
                                        output_id=new_output_id,
                                        read_frame=vertex_pair.read_frame.read_phase,
                                        read_frame_annotated=vertex_pair.read_frame.annotated_RF,
                                        gene_name=gene.name,
                                        gene_chr=gene.chr,
                                        gene_strand=gene.strand,
                                        mutation_mode=mutation.mode,
                                        has_stop_codon=int(flag.has_stop),
                                        is_in_junction_list=is_intron_in_junction_list_flag,
                                        is_isolated=int(flag.is_isolated),
                                        variant_comb=variant_comb,
                                        variant_seg_expr=seg_exp_variant_comb,
                                        modified_exons_coord=modi_coord,
                                        original_exons_coord=vertex_pair.original_exons_coord,
                                        vertex_idx=vertex_list,
                                        kmer_type=kmer_type
                                        ), sep = '\t'))
                        variant_id += 1
                        output_peptide = OutputPeptide(output_id=new_output_id,
                                                        peptide=peptide.mut[pep_idx],
                                                        exons_coor=modi_coord,
                                                        strand=gene.strand,
                                                        read_frame_annotated=vertex_pair.read_frame.annotated_RF)

                        ### kmers
                        # Calculate the output kmer and the corresponding expression based on output peptide
                        create_kmer(output_peptide, kmer, gene_kmer_coord, kmer_database) 

                        if len(peptide_set) > len_pep_save: # Save peptide batch to disk when threshold is reached
                            save_fg_peptide_set(peptide_set, filepointer, out_dir, fasta_save,
                                                verbose=False, id_tag=f'{kmer_type}{ii}')
                            peptide_set.clear()

            if not gene.splicegraph.edges is None:
                gene.to_sparse()
                
        #TODO: change all below 
        prepare_output_kmer(gene, idx, countinfo, seg_counts, edge_idxs, edge_counts,
                            gene_kmer_coord, gene_annot_jx,
                            graph_output_samples_ids,
                            graph_samples, filepointer, out_dir, verbose=verbose_save)
        # Save the last batch of peptides
        save_fg_peptide_set(peptide_set, filepointer, out_dir, fasta_save,
                            verbose=False, id_tag=f'{kmer_type}{ii}')