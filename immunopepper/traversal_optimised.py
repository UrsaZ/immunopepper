from collections import deque
from typing import List, Tuple, Dict, Set, Any
import numpy as np

# Define stop codons for early termination
#TODO: alternatively use cds coordinate
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Main function to extract unique k-mers
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

    # Pull graph components
    segments = gene.segmentgraph.segments
    seg_edges = gene.segmentgraph.seg_edges

    # Identify terminal segments (those without any outgoing edges)
    n_segments = seg_edges.shape[0]
    terminal_segments = set(np.where(np.sum(seg_edges, axis=1) == 0)[0])

    # Set to store final k-mers as tuples of (segment_id, start, end)
    unique_kmers: Set[Tuple[Tuple[int, int, int], ...]] = set()

    # Queue for k-mers to be propagated 
    # deque is a list-like container with fast appends and pops on either end
    queue: deque = deque()

    # Initialize 27-mers from CDS start positions
    for seg_id, offset in cds_start_coords:
        init_paths = build_initial_kmers(seg_id, offset, k, segment_sequences, seg_edges)
        for path in init_paths:
            queue.append(path) #TODO: ensure initial path is non-redundant!
            unique_kmers.add(tuple(path))


    # Propagate k-mers (active paths) through the segment graph
    # the graph is traversed 5'--> 3', not transcript by transcript.
    while queue: # While not all terminal k-mers are processed
        current_path = queue.popleft()  # Remove and return a k-mer from the left side

        # Try to advance by 3 nt (--> 1 aa)
        new_path = propagate_kmer(current_path, step, segment_sequences, seg_edges, terminal_segments)

        if new_path is not None: # if path did not end
            # Check for stop codon in the new last 3 nt, terminate path if found
            if not contains_stop_codon(new_path, segment_sequences):
                path_tuple = tuple(new_path)
                # e.g. will be true for alternative starts, which all lead to the same segment
                # this segment needs to be propagated only once, so we do not append it to queue again
                if path_tuple not in unique_kmers:
                    unique_kmers.add(path_tuple)
                    queue.append(new_path)

    return unique_kmers


def build_initial_kmers(seg_id: int,
                        offset: int,
                        k: int,
                        segment_sequences: Dict[int, str],
                        seg_edges: np.ndarray) -> List[List[Tuple[int, int, int]]]:
    """
    Build all valid initial k-mer paths starting from (seg_id, offset), exploring
    all splice paths until k nucleotides are collected.

    Returns:
        List of paths, each path is a list of (segment_id, start, end) tuples
    """

    results = []

    def dfs(current_id: int,
            current_offset: int,
            remaining: int,
            path: List[Tuple[int, int, int]]):
        """
        Recursive DFS to explore all segment paths collecting exactly k nt.
        """
        seq = segment_sequences.get(current_id, "")
        if current_offset >= len(seq):
            return  # Invalid start point

        # Calculate how many nt we can take from this segment
        available = len(seq) - current_offset
        take = min(available, remaining)

        # Extend path
        new_path = path + [(current_id, current_offset, current_offset + take)]
        remaining -= take

        if remaining == 0:
            results.append(new_path)
            return

        # Need more nt: follow all outgoing edges
        next_segments = np.where(seg_edges[current_id])[0]  #FIXME: what if no next segments?
        for next_id in next_segments:
            dfs(next_id, 0, remaining, new_path)

    # Start DFS from the initial segment and offset
    dfs(seg_id, offset, k, [])

    return results



def propagate_kmer(path: List[Tuple[int, int, int]],
                   step: int,
                   segment_sequences: Dict[int, str],
                   seg_edges: np.ndarray,
                   terminal_segments: Set[int]) -> List[Tuple[int, int, int]] | None:
    """
    Propagate a k-mer forward by 'step' nt. Returns a new path or None if path ends.
    """
    new_path = [] # Save new path here
    to_shift = step # How many nt we need to shift the start of the path (start with 3)
    idx = 0 # Start from the first segment in the path

    # Shift the start of the path by 3 nt
    while to_shift > 0 and idx < len(path): # while we still have to shift and there are segments in the path
        seg_id, start, end = path[idx]  # Get 1st segment in the path info
        seg_len = end - start
        if seg_len <= to_shift: # if the segment is shorter than the shift
            to_shift -= seg_len # remaning will be subtracted from the next segment
            idx += 1 # move to the next segment
        else: # if the segment is long enough to shift
            new_path = path[idx:] # Copy the rest of the path
            new_path[0] = (seg_id, start + to_shift, end) # modify the start of the current segment
            break
    else: # If we finished the loop without breaking --> to_shift == 0 and it was done by seg_len == to_shift
        new_path = path[idx:]

    if not new_path:
        return None  # Nothing left

    # Extend the end of the k-mer by 3 nt
    last_seg_id, last_start, last_end = new_path[-1]
    seq = segment_sequences[last_seg_id]
    available = len(seq) - last_end # Check if the last segment has enough nt-s to extend

    if available >= step: # Enough nt-s in the current segment
        new_path[-1] = (last_seg_id, last_start, last_end + step) # Modify the end of the last segment
    else:
        # Extend into the next segment
        remaining = step - available
        new_path[-1] = (last_seg_id, last_start, len(seq)) # Extend to the very end of the last segment

        next_segments = np.where(seg_edges[last_seg_id])[0] # Get all possible downstream segments
        if len(next_segments) == 0:
            return None  # No downstream segments #TODO: check if ok this short k-mer is not saved

        # Try extending into any downstream segment (could be forked in full version)
        #FIXME: what about segments shorter than 3 nt?
        for next_id in next_segments:
            next_seq = segment_sequences.get(next_id, "")
            if len(next_seq) >= remaining:
                new_path.append((next_id, 0, remaining))
                break
        else:
            return None  # Can't extend

    # Terminate if last segment is terminal
    if new_path[-1][0] in terminal_segments:
        return None

    return new_path


def contains_stop_codon(path: List[Tuple[int, int, int]],
                        segment_sequences: Dict[int, str]) -> bool:
    """
    Check whether a stop codon exists in the sequence represented by the path.
    """
    seq = ""
    for seg_id, start, end in path: #TODO: enough to check last 5 nt?
        seq += segment_sequences[seg_id][start:end]

    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon in STOP_CODONS:
            return True

    return False