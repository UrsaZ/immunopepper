from collections import deque
from typing import List, Tuple, Dict, Set
import numpy as np
import logging

# Define stop codons for early termination
STOP_CODONS = {"TAA", "TAG", "TGA"}

# A trie class to hold all valid segment paths derived from real transcripts
class SegmentPathTrie:
    def __init__(self):
        self.root = {}

    def insert(self, path: List[int]):
        node = self.root
        for seg_id in path:
            if seg_id not in node:
                node[seg_id] = {}
            node = node[seg_id]
        node['__END__'] = True

    def is_valid_path(self, path: List[int]) -> bool:
        """
        Return True for a segment id path that matches a full path in the trie. 
        """
        node = self.root
        for seg_id in path:
            if seg_id in node:
                node = node[seg_id]
            else:
                return False
        return True
    
    def children(self, path: List[int]) -> List[int]:
        """
        Return all valid next segment IDs that follow the given segment path prefix,
        sorted in ascending order.
        """
        node = self.root
        for seg_id in path:
            if seg_id in node:
                node = node[seg_id]
            else:
                return []
        return sorted(k for k in node.keys() if k != '__END__')
    
    def get_all_paths(self) -> List[List[int]]:
        def dfs(node, path, paths):
            for key, child in node.items():
                if key == '__END__':
                    paths.append(path[:])
                else:
                    path.append(key)
                    dfs(child, path, paths)
                    path.pop()
        all_paths = []
        dfs(self.root, [], all_paths)
        return all_paths

    def __str__(self):
        paths = self.get_all_paths()
        return '\n'.join(f"Path {i+1}: {path}" for i, path in enumerate(paths))

def build_segment_trie(gene) -> SegmentPathTrie:
    """
    Build a trie of valid segment paths by traversing the splicegraph.
    Segment paths are derived from exon connectivity and segment-exon matches.
    If the strand is '-', the path is reversed at the end.

    Args:
        gene: An object containing the strand, splicegraph and segmentgraph of a gene.

    Returns:
        SegmentPathTrie: A trie containing all valid segment paths derived from splicegraph paths.
    """
    trie = SegmentPathTrie()
    seg_match = gene.segmentgraph.seg_match # 2D boolean matrix (exons x segments) 
    splice_edges = gene.splicegraph.edges # 2D boolean adjacency matrix (exons x exons)
    exon_coords = gene.splicegraph.vertices.T  # (N, 2) → [start, end] per exon
    # a dict mapping each exon ID to the list of segment IDs it contains
    exon_to_segments = {
        exon_id: list(np.where(seg_match[exon_id])[0])
        for exon_id in range(seg_match.shape[0])
    }

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
            trie.insert(segment_path) # Insert the complete segment path into the trie
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

    return trie

def extract_kmers_from_graph(gene,
                             segment_sequences: Dict[int, str],
                             cds_start_coords: List[Tuple[int, int]],
                             k: int = 27,
                             step: int = 3) -> Set[Tuple[Tuple[int, int, int], ...]]:
    """
    Extract unique k-mers from a segment graph using propagation strategy.

    Parameters:
        gene: Gene object with .segmentgraph and .splicegraph
        segment_sequences: dict mapping segment_id -> DNA sequence (str)    #TODO: can use ref chromosome sequence here
        cds_start_coords: list of tuples (segment_id, offset) marking CDS starts #TODO: cds list of cds start coordinates
        k: k-mer size (default 27)
        step: propagation step in nt (default 3)

    Returns:
        Set of unique k-mers, each as a tuple of (segment_id, start, end)
    """

    # Build a trie of valid segment paths from actual transcripts
    trie = build_segment_trie(gene)

    # Set to store final k-mers as tuples of (segment_id, start, end)
    unique_kmers: Set[Tuple[Tuple[int, int, int], ...]] = set()

    # Queue for k-mers to be propagated 
    # deque is a list-like container with fast appends and pops on either end
    queue: deque = deque()

    # Initialize 27-mers from CDS start positions
    for seg_id, offset in cds_start_coords:
        init_paths = build_initial_kmers(seg_id, offset, k, segment_sequences, gene.segmentgraph.seg_edges, trie)
        for path in init_paths:
            queue.append(path)
            unique_kmers.add(tuple(path))

    # Propagate k-mers (active paths) through the segment graph
    # the graph is traversed 5'--> 3', not transcript by transcript.
    while queue: # While there are k-mers to propagate
        current_path = queue.popleft() # Remove and return a k-mer from the left side
        
        # Try to advance by 3 nt (--> 1 aa)
        # new_paths is a list of lists of tuples (segment_id, start, end)
        new_paths = propagate_kmer(current_path, step, segment_sequences, gene.segmentgraph.seg_edges, gene.segmentgraph.segments, trie)

        # iterate over all possible propagated paths
        for new_path in new_paths:
            if not contains_stop_codon(new_path, segment_sequences): # Check for stop codon in the new last 3 nt, terminate path if found
                path_tuple = tuple(new_path)

                # this will be true for alternative starts, which all lead to the same segment
                # this segment needs to be propagated only once, so we do not append it to queue again
                if path_tuple not in unique_kmers:
                    unique_kmers.add(path_tuple)
                    queue.append(new_path)

    return unique_kmers

def build_initial_kmers(cds_starts: List[int],
                        k: int,
                        segments: np.ndarray,
                        strand: str,
                        trie: SegmentPathTrie) -> List[List[Tuple[int, int, int]]]:
    """
    Build all valid k-mer paths (e.g., 27-mers) starting from CDS start positions.

    Args:
        cds_starts: List of genomic CDS start coordinates.
        k: k-mer length (number of nucleotides).
        segments: 2xN array of genomic start and end positions per segment (gene.segmentgraph.segments).
        strand: '+' or '-' indicating gene orientation.
        trie: A SegmentPathTrie used to validate segment paths.

    Returns:
        List of k-mer paths. Each path is a list of (segment_id, genomic_start, genomic_end) tuples.
    """
    results = []
    seen = set()

    # Build a lookup for fast start coordinate → (segment_id, offset) resolution
    seg_coords = segments.T  # Shape (N, 2), each row: (start, end)
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

        # Propagate to all valid children in the trie
        next_seg_ids = trie.children(seg_path)
        for next_id in next_seg_ids:
            next_seg_path = seg_path + [next_id]
            if trie.is_valid_path(next_seg_path):
                dfs(next_id, 0, remaining, new_path, next_seg_path)

    # Start DFS from each CDS start position
    for cds_start in cds_starts:
        seg_id, offset = find_segment_and_offset(cds_start)
        if seg_id is not None:
            dfs(seg_id, offset, k, [], [seg_id])

    return results

def propagate_kmer(path: List[Tuple[int, int, int]],
                   step: int,
                   segment_sequences: Dict[int, str],
                   seg_edges: np.ndarray,
                   segments: np.ndarray,
                   trie: SegmentPathTrie) -> List[List[Tuple[int, int, int]]]:
    """
    Propagate a k-mer forward by 'step' nt in all valid directions, constrained by the SegmentPathTrie.

    Args:
        path: Current k-mer path as a list of (segment_id, start, end).
        step: Number of nt to shift forward.
        segment_sequences: Segment ID -> sequence string.
        seg_edges: Segment adjacency matrix (spladder gene.segmentgraph.seg_edges).
        segments: 2xN with [start, end] for each segment based on the strand (spladder gene.segmentgraph.segments) #TODO: test with both strands
        trie: A SegmentPathTrie object that validates segment paths.

    Returns:
        A list of all possible propagated k-mer paths (each a list of tuples).
    """
    all_paths = [] # Will hold all valid propagated paths

    # Compute segment path inline (list of segment IDs in the current path)
    # used to check if the next segment is connected to the path
    seg_path = [s for s, _, _ in path]

    # --- Step 1: Shift the start of the path forward by 'step' nt ---
    to_shift = step # How many nt we need to shift the start of the path (start with 3)
    idx = 0 # Which segment we are currently processing in the path
    shifted_path = [] # Will hold the path with shifted START

    while to_shift > 0 and idx < len(path): # while we still have to shift and the current segment (idx) exists
        seg_id, start, end = path[idx]  # Get current segment info
        seg_len = end - start
        if seg_len <= to_shift: # if the segment is shorter than the shift
            to_shift -= seg_len # remaning will be subtracted from the next segment
            idx += 1 # move to the next segment
        else: # if the segment is long enough to shift
            shifted_path = path[idx:] # Copy the rest of the path
            shifted_path[0] = (seg_id, start + to_shift, end) # modify the start of the current segment
            break
    else: # The loop finishes without breaking if to_shift == 0 and it was done by seg_len == to_shift
        shifted_path = path[idx:]

    if not shifted_path:
        return []

    # --- Step 2: Recursively extend the path's end by 'step' nt ---
    def extend_path(current_path: List[Tuple[int, int, int]],
                    current_seg_path: List[int],
                    step_remaining: int):
        """
        Recursively extend the current k-mer path by a given number of nucleotides (step_remaining),
        traversing into downstream segments as needed. Only valid segment paths are allowed,
        as determined by the SegmentPathTrie.

        Args:
            current_path: List of (segment_id, start, end) tuples representing the current (partial) k-mer path.
            current_seg_path: List of segment IDs in the current_path, used to validate paths in the trie.
            step_remaining: Number of nucleotides still needed to complete the propagation step.

        Side Effects:
            Appends all valid extended paths to the `all_paths` list.
        """

        last_seg_id, last_start, last_end = current_path[-1]
        seq = segment_sequences.get(last_seg_id, "")
        available = len(seq) - last_end

        if available >= step_remaining:
            # Extend within current segment
            extended = current_path[:-1] + [(last_seg_id, last_start, last_end + step_remaining)] # concatenate two lists
            all_paths.append(extended)
        else:
            # Extend to end of current segment
            extended_partial = current_path[:-1] + [(last_seg_id, last_start, len(seq))]
            remaining = step_remaining - available

            next_segments = np.where(seg_edges[last_seg_id])[0] # Get all segments connected to the last segment
            # Iterate over all possible next segments
            for next_id in next_segments:
                new_seg_path = current_seg_path + [next_id] # add next segment to the path (a list of segment IDs)
                if not trie.is_valid_path(new_seg_path): # check if this path exists in any transcript
                    continue  # Skip invalid transitions

                next_seq = segment_sequences.get(next_id, "")
                if not next_seq:
                    continue

                take = min(len(next_seq), remaining) # take the remaining if possible, otherwise take the whole segment length
                seg_start = segments[0, next_id] # get next segment start position from gene.segmentgraph.segments
                new_path = extended_partial + [(next_id, seg_start, seg_start + take)]

                if take == remaining: # only if we can take the whole remaining length
                    all_paths.append(new_path)
                else: # still have remaining length to propagate
                    extend_path(new_path, new_seg_path, remaining - take)

    extend_path(shifted_path, seg_path, step) # actual call to extend the path; shifted_path is the path with shifted start
    #TODO: update seg_path after shifting the start

    return all_paths

#TODO: optimise in the end, no need to check the whole path after every propagation
def contains_stop_codon(path: List[Tuple[int, int, int]],
                        segment_sequences: Dict[int, str]) -> bool:
    seq = ""
    for seg_id, start, end in path:
        seq += segment_sequences[seg_id][start:end]

    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon in STOP_CODONS:
            return True

    return False