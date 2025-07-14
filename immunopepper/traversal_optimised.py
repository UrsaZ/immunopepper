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
        node = self.root
        for seg_id in path:
            if seg_id in node:
                node = node[seg_id]
            else:
                return False
        return True

def build_segment_trie(gene) -> SegmentPathTrie:
    """
    Build a trie of valid segment paths from actual transcripts.
    Each transcript is represented by a sequence of exons,
    and each exon is mapped to its corresponding segment IDs.

    Args:
        gene: An object containing the splicegraph and segmentgraph of a gene.

    Returns:
        SegmentPathTrie: A trie containing all valid segment paths derived from transcripts.
    """

    seg_match = gene.segmentgraph.seg_match  # shape: (n_exons, n_segments)
    # gene.splicegraph.vertices: 2xN array where each column represents an exon: [start, end] based on the strand
    vertices_list = gene.splicegraph.vertices.T.tolist()  # transpose to [start, end] rows

    exon_to_segments = {
        exon_id: set(np.where(seg_match[exon_id])[0])
        for exon_id in range(seg_match.shape[0])
    } # Map exon IDs to their corresponding segment IDs

    trie = SegmentPathTrie()

    for transcript_idx in gene.transcripts: # get exons for each transcript
        segment_path = []
        exon_coords = gene.exons[transcript_idx]  # List of (start, end) tuples as ndarray

        for exon_start, exon_end in exon_coords: # get exon id from exon coordinates
            try:
                exon_id = vertices_list.index([exon_start, exon_end])
            except ValueError: # exon not found in splicegraph
                logging.warning(f"Exon {exon_start}-{exon_end} not found in splicegraph.")
                continue

            segment_path.extend(sorted(exon_to_segments[exon_id]))

        trie.insert(segment_path)

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
        segment_sequences: dict mapping segment_id -> DNA sequence (str)    #TODO: no need for sequences if I apply cds stops instead of stop codons
        cds_start_coords: list of tuples (segment_id, offset) marking CDS starts #TODO: find out how cds are encoded
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
            queue.append((path, [seg_id])) #TODO: ensure initial path is non-redundant!
            unique_kmers.add(tuple(path))

    # Propagate k-mers (active paths) through the segment graph
    # the graph is traversed 5'--> 3', not transcript by transcript.
    while queue: # While there are k-mers to propagate
        current_path, seg_path = queue.popleft() # Remove and return a k-mer from the left side
        
        # Try to advance by 3 nt (--> 1 aa)
        # new_paths is a list of lists of tuples (segment_id, start, end)
        new_paths = propagate_kmer(current_path, step, segment_sequences, gene.segmentgraph.seg_edges, gene.segmentgraph.segments, trie)

        for new_path in new_paths:
            # If the segment ID at the end of the new path is different from the last segment ID in the current path, then append it to the path
            new_seg_path = seg_path + [new_path[-1][0]] if new_path[-1][0] != seg_path[-1] else seg_path
            if not contains_stop_codon(new_path, segment_sequences):
                path_tuple = tuple(new_path)
                if path_tuple not in unique_kmers:
                    unique_kmers.add(path_tuple)
                    queue.append((new_path, new_seg_path))

        if new_path is not None: # if the path did not end
            new_seg_path = seg_path + [new_path[-1][0]] if new_path[-1][0] != seg_path[-1] else seg_path #?
            
            # check if the new segment combination exists in any transcript
            if trie.is_valid_path(new_seg_path):
                if not contains_stop_codon(new_path, segment_sequences): # terminate if stop codon is found
                    path_tuple = tuple(new_path) 

                    # will be true for alternative starts, which all lead to the same segment
                    # this segment needs to be propagated only once, so we do not append it to queue again
                    if path_tuple not in unique_kmers:
                        unique_kmers.add(path_tuple)
                        queue.append((new_path, new_seg_path))

    return unique_kmers

def build_initial_kmers(seg_id: int,
                        offset: int,
                        k: int,
                        segment_sequences: Dict[int, str],
                        seg_edges: np.ndarray,
                        trie: SegmentPathTrie) -> List[List[Tuple[int, int, int]]]:
    results = []

    def dfs(current_id: int, current_offset: int, remaining: int,
            path: List[Tuple[int, int, int]], seg_path: List[int]):
        seq = segment_sequences.get(current_id, "")
        if current_offset >= len(seq):
            return

        available = len(seq) - current_offset
        take = min(available, remaining)
        new_path = path + [(current_id, current_offset, current_offset + take)]
        new_seg_path = seg_path + [current_id] if not seg_path or seg_path[-1] != current_id else seg_path
        remaining -= take

        if remaining == 0:
            results.append(new_path)
            return

        for next_id in np.where(seg_edges[current_id])[0]:
            if trie.is_valid_path(new_seg_path + [next_id]):
                dfs(next_id, 0, remaining, new_path, new_seg_path)

    dfs(seg_id, offset, k, [], [])
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