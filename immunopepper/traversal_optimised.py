from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set
import numpy as np
import logging

from immunopepper.namedtuples import GeneTable

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
    """
    results = []
    seen = set()

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

def extract_sequence_from_coords_int(coords: List[int], 
                                 gene_sequence: str, 
                                 gene_start: int) -> str:
    """
    Extracts a nucleotide sequence from a gene sequence using a list of genomic coordinates.

    Args:
        coords: List of genomic coordinates of the last three NTs.
        gene_sequence: The nucleotide sequence corresponding to the full gene region.
        gene_start: Genomic start coordinate of the gene (used to align coords to gene_sequence).

    Returns:
        str: The concatenated nucleotide sequence for all provided coordinate segments.
    """
    seq = [gene_sequence[abs_c - gene_start] for abs_c in coords]
    return ''.join(seq)

def extract_sequence_from_coords_tuple(coords: List[Tuple[int, int]], 
                                 gene_sequence: str, 
                                 gene_start: int) -> str:
    """
    Extracts a nucleotide sequence from a gene sequence using a list of genomic coordinates.

    Args:
        coords: List of (seg_id, start, end) genomic coordinate tuples.
        gene_sequence: The nucleotide sequence corresponding to the full gene region.
        gene_start: Genomic start coordinate of the gene (used to align coords to gene_sequence).

    Returns:
        str: The concatenated nucleotide sequence for all provided coordinate segments.
    """
    seq = []
    sorted_coords = sorted(coords) # make sure the segments are in the increasing order
    for (seg_id, start, end) in sorted_coords:
        rel_start = start - gene_start
        rel_end = end - gene_start
        seq.append(gene_sequence[rel_start:rel_end])
    return ''.join(seq)

#TODO: instead of my custom functions, try using existing functions/Biophython
def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]

def check_stop_codon(seq: str) -> bool:
    return seq[-3:].upper() in {"TAA", "TAG", "TGA"} # check last three NT

def extract_kmers_from_graph(gene,
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
    # Get gene's sequence (str). If no germline mutations, use the reference sequence.
    seq = ref_mut_seq['background']

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
    #TODO: check for stop codon!
    init_paths = build_initial_kmers(cds_starts, k, gene.segmentgraph.segments, gene.strand, index)
    for path in init_paths:
        queue.append(path)
        unique_kmers.add(tuple(path))

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
                
                # coordinates --> NTs
                nt_seq = extract_sequence_from_coords_tuple(new_path, seq, gene.start) #TODO: check if this could be done more efficiently
                unique_kmers.add(path_tuple)

                if stop_on_stop:
                    if check_stop_codon(nt_seq):
                        continue  # stop codon found → don’t propagate further
                queue.append(new_path)  # no stop codon → continue propagating

    return unique_kmers
