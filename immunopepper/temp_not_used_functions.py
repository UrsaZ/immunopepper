from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Union, Optional
import numpy as np
import logging
import itertools

from immunopepper.dna_to_peptide import dna_to_peptide
from immunopepper.namedtuples import GeneTable, Coord, Flag, Peptide
from immunopepper.translate import complementary_seq, get_peptide_result
from immunopepper.filter import is_intron_in_junction_list
#from immunopepper.traversal_optimised import build_initial_kmers, propagate_kmer
from immunopepper.mutations import get_mut_comb, get_mutated_sequence

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

def extract_sequence_from_kmers(coords: List[Tuple[int, int, int]], 
                                 gene_sequence: str, 
                                 gene_start: int, 
                                 strand: str) -> str:
    """
    Extracts a nucleotide sequence from a gene sequence using a list of genomic coordinates.

    Args:
        coords: List of (seg_id, start, end) genomic coordinate tuples.
        gene_sequence: The nucleotide sequence corresponding to the full gene region.
        gene_start: Genomic start coordinate of the gene (used to align coords to gene_sequence).
        strand: '+' or '-' indicating the strand of the gene.

    Returns:
        str: The (possibly reverse complemented) nucleotide sequence for the kmer.
    """
    seq = []
    sorted_coords = sorted(coords) #TODO: ask if this OK!

    for (seg_id, start, end) in sorted_coords:
        rel_start = start - gene_start
        rel_end = end - gene_start
        seq.append(gene_sequence[rel_start:rel_end])

    joined = ''.join(seq)
    if strand == '-':
        return reverse_complement(joined)
    return joined

def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]

def check_stop_codon(seq: str) -> bool:
    return seq[-3:].upper() in {"TAA", "TAG", "TGA"} # check last three NT

def has_in_frame_stop(seq: str) -> Tuple[bool, Optional[int]]:
    """
    Checks for the presence of an in-frame stop codon anywhere in the kmer.

    Args:
        seq: A nucleotide sequence (expected to be in-frame)

    Returns:
        (has_stop: bool, position: Optional[int]) where position is the
        index of the first nucleotide of the stop codon, or None if not found.
    """
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon.upper() in {"TAA", "TAG", "TGA"}:
            return True, i  # i is the relative NT position of the stop
    return False, None

def get_kmers_and_translate(gene,
                             ref_mut_seq: str,
                             genetable: GeneTable,
                             sub_mutation: dict,
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
    seg_match = gene.segmentgraph.seg_match  # 2D boolean matrix (exons x segments) 
    exon_to_segments = {
        exon_id: list(np.where(seg_match[exon_id])[0])
        for exon_id in range(seg_match.shape[0])}
    index = build_segment_index(gene, exon_to_segments)

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
        mut_seq_comb = get_mut_comb(path, sub_mutation.somatic_dict)
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
                mut_seq_comb = get_mut_comb(path, sub_mutation.somatic_dict)
                for variant_comb in mut_seq_comb:
                    peptide, flag = get_peptide_result(new_path, gene.strand, variant_comb, sub_mutation.somatic_dict, ref_mut_seq, gene.start)
                    
                    if stop_on_stop:
                        if flag.has_stop:
                            continue
                    if new_path not in queue:
                        queue.append(new_path)  # no stop codon → continue propagating

    return unique_kmers

class SegmentPathTrie:
    def __init__(self):
        self.root = {}
        self.transitions = defaultdict(set)  # {from_seg_id: set(of next seg_ids)}

    def insert(self, path: List[int]):
        node = self.root
        for i in range(len(path)):
            seg_id = path[i]
            if seg_id not in node:
                node[seg_id] = {}
            if i + 1 < len(path):
                next_seg_id = path[i + 1]
                self.transitions[seg_id].add(next_seg_id)
            node = node[seg_id]
        node['__END__'] = True

    def children(self, partial_path: List[int]) -> List[int]:
        """
        Return valid next segment IDs that follow the given partial segment path,
        even if the path starts inside any full path stored in the trie.

        For example, if the trie contains:
            - [3, 2, 1, 0]
            - [3, 1, 0]

        Then:
            - partial_path [2, 1] → [0]
            - partial_path [1]    → [0]
            - partial_path [3]    → [2, 1]
        """
        results = []

        def dfs(node, current_path):
            for seg_id, child in node.items():
                if seg_id == '__END__':
                    continue
                new_path = current_path + [seg_id]
                if new_path[-len(partial_path):] == partial_path:
                    # If partial path matches tail, collect next children
                    results.extend(k for k in child.keys() if k != '__END__')
                dfs(child, new_path)

        dfs(self.root, [])
        return sorted(set(results))


    def get_all_paths(self) -> List[List[int]]:
        """Get all complete paths stored in the trie for visualization or debugging."""
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
