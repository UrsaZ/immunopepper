"""Countains code related to translation"""

import logging
import numpy as np
#TODO For developement
import pyximport; pyximport.install()
import sys
from typing import List, Tuple, Dict, Set

from immunopepper.dna_to_peptide import dna_to_peptide

from immunopepper.namedtuples import Coord, Flag, Kmer, Peptide, ReadingFrameTuple
from immunopepper.utils import get_exon_expr, get_sub_mut_dna

def get_full_peptide(gene, seq, cds_list, countinfo, seg_counts=None, Idx=None, all_read_frames=False):
    """
    Output translated peptide and segment expression list given cds_list

    Parameters
    ----------
    gene: Object, created by SplAdder.
    seq: str. Gene sequence.
    cds_list: List[Tuple(v_start,v_stop,reading_frame)]
    countinfo: Namedtuple, SplAdder count information
    seg_counts: np.array, array of spladder segment counts from gene and sample of interest
    Idx: Namedtuple Idx, has attribute idx.gene and idx.sample

    Returns
    -------
    cds_expr_list: List[Tuple(segment_length,segment_expression)]
    cds_string: str. Concatenated sequence string according to cds_list
    cds_peptide: str. Translated peptide string according to cds_list

    """
    if gene.strand.strip() == "-":
        cds_list = cds_list[::-1]
    gene_start = np.min(gene.splicegraph.vertices)

    cds_string = ""
    first_cds = True
    cds_expr_list = []
    # Append transcribed CDS regions to the output
    for coord_left, coord_right, frameshift in cds_list:

        # Apply initial frameshift on the first CDS of the transcript
        if first_cds:
            if gene.strand.strip() == "+":
                coord_left += frameshift
            else:
                coord_right -= frameshift
            first_cds = False
        cds_expr = get_exon_expr(gene, coord_left, coord_right, countinfo, Idx, seg_counts)
        cds_expr_list.extend(cds_expr)
        nuc_seq = seq[coord_left - gene_start:coord_right - gene_start]

        # Accumulate new DNA sequence...
        if gene.strand.strip() == "+":
            cds_string += nuc_seq
        elif gene.strand.strip() == "-":
            cds_string += complementary_seq(nuc_seq[::-1])
        else:
            logging.error("ERROR: Invalid strand. Got %s but expect + or -" % gene.strand.strip())
            sys.exit(1)
    cds_peptide, is_stop_flag = dna_to_peptide(cds_string, all_read_frames)
    return cds_expr_list, cds_string, cds_peptide


def complementary_seq(dna_seq):
    """ Yields the complementary DNA sequence
    Only convert the character in comp_dict.
    Otherwise remain the same.
    """
    comp_dict = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([comp_dict[nuc] if nuc in comp_dict else nuc for nuc in dna_seq])


def cross_peptide_result(read_frame, strand, variant_comb, somatic_mutation_sub_dict, ref_mut_seq, peptide_accept_coord, gene_start, all_read_frames):
    """ Get translated peptide from the given exon pairs.

    Parameters
    ----------
    read_frame: NamedTuple. (read_start_codon, read_stop_codon, emitting_frame)
    strand: str. '+' or '-'
    variant_comb: List(int).
    somatic_mutation_sub_dict: Dict. variant position -> variant details.
    ref_mut_seq: Dict.['ref', 'background'] -> List(str)
    peptide_accept_coord: The start and end position of next vertex. Positions of the first vertex
        are already given in read_frame.

    Returns
    -------
    peptide: NamedTuple Peptide. has attribute ['ref', 'mut']. contain the output peptide
        translated from reference sequence and mutated sequence.
    coord: NamedTuple Coord. has attribute ['start_v1', 'stop_v1', 'start_v2', 'stop_v2']
        contains the true four position of exon pairs (after considering read frame)
        that outputs the peptide.
    flag: NamedTuple Flag. has attribute ['has_stop', 'is_isolated']
    next_start_v1: int. start vertex in new reading frame
    next_stop_v1: int. stop vertex in new reading frame
    next_emitting_frame: int. Shift induced

    """
    cds_left_modi, cds_right_modi, emitting_frame = read_frame[0], read_frame[1], read_frame[2]
    next_emitting_frame = (peptide_accept_coord[1] - peptide_accept_coord[0] + emitting_frame) % 3
    start_v1 = cds_left_modi
    stop_v1 = cds_right_modi

    #                                       |next_start_v1  |
    # |      v1           | |    v2                         |
    # -----[emitting_frame] [accepting_frame]-------
    # emitting_frame + accepting_frame = 3
    accepting_frame = (3 - emitting_frame) % 3

    if somatic_mutation_sub_dict:  # exist maf dictionary, so we use germline mutation-applied seq as the background seq
        ref_seq = ref_mut_seq['background']
    else:
        ref_seq = ref_mut_seq['ref']
    mut_seq = ref_mut_seq['background']
    # python is 0-based while gene annotation file(.gtf) is one based
    # so we need to do a little modification
    if strand == "+":
        start_v2 = peptide_accept_coord[0]
        stop_v2 = max(start_v2, peptide_accept_coord[1] - next_emitting_frame)
        coord = Coord(start_v1, stop_v1, start_v2, stop_v2)
        peptide_dna_str_mut = get_sub_mut_dna(mut_seq, coord, variant_comb, somatic_mutation_sub_dict, strand, gene_start)
        peptide_dna_str_ref = ref_seq[start_v1 - gene_start:stop_v1 - gene_start] + ref_seq[start_v2 - gene_start:stop_v2 - gene_start]
        next_start_v1 = min(start_v2 + accepting_frame, peptide_accept_coord[1])
        next_stop_v1 = peptide_accept_coord[1]
    else:  # strand == "-"
        stop_v2 = peptide_accept_coord[1]
        start_v2 = min(stop_v2, peptide_accept_coord[0] + next_emitting_frame)
        coord = Coord(start_v1, stop_v1, start_v2, stop_v2)
        peptide_dna_str_mut = complementary_seq(get_sub_mut_dna(mut_seq, coord, variant_comb, somatic_mutation_sub_dict, strand, gene_start))
        peptide_dna_str_ref = complementary_seq(ref_seq[start_v1 - gene_start:stop_v1 - gene_start][::-1] + ref_seq[start_v2 - gene_start:stop_v2 - gene_start][::-1])
        next_start_v1 = peptide_accept_coord[0]
        next_stop_v1 = max(stop_v2 - accepting_frame, peptide_accept_coord[0])

    assert (len(peptide_dna_str_mut) == len(peptide_dna_str_ref))
    # if len(peptide_dna_str_mut) % 3 != 0:
    #     print("Applied mutations have changed the length of the DNA fragment - no longer divisible by 3")
    peptide_mut, mut_has_stop_codon = dna_to_peptide(peptide_dna_str_mut, all_read_frames)
    peptide_ref, ref_has_stop_codon = dna_to_peptide(peptide_dna_str_ref, all_read_frames)

    # if the stop codon appears before translating the second exon, mark 'single'
    is_isolated = False
    if len(peptide_mut[0])*3 <= abs(stop_v1 - start_v1) + 1:
        is_isolated = True
        jpos = 0.0
    else:
        jpos = float(stop_v1 - start_v1) / 3.0
    peptide = Peptide(peptide_mut, peptide_ref)
    coord = Coord(start_v1, stop_v1, start_v2, stop_v2)
    flag = Flag(mut_has_stop_codon, is_isolated)
    return peptide, coord, flag, next_start_v1, next_stop_v1, next_emitting_frame


def isolated_peptide_result(read_frame, strand, variant_comb, somatic_mutation_sub_dict, ref_mut_seq, gene_start, all_read_frames):
    """ Deal with translating isolated cds, almost the same as cross_peptide_result

    Parameters
    ----------
    read_frame: Tuple. (read_start_codon, read_stop_codon, emitting_frame)
    strand: str. '+' or '-'
    variant_comb: List(int).
    somatic_mutation_sub_dict: Dict. variant position -> variant details.
    ref_mut_seq: Dict.['ref', 'background'] -> List(str)

    Returns
    -------
    peptide: NamedTuple. has attribute ['ref', 'mut']. contain the output peptide
        translated from reference sequence and mutated sequence.
    coord: NamedTuple. has attribute ['start_v1', 'stop_v1', 'start_v2', 'stop_v2']
        contains the true two position of exon pairs (after considering read frame)
        that outputs the peptide. 'start_v2', 'stop_v2' is set to be np.nan.
    flag: NamedTuple. has attribute ['has_stop', 'is_isolated']

    """

    start_v1 = read_frame.cds_left_modi
    stop_v1 = read_frame.cds_right_modi
    emitting_frame = read_frame.read_phase
    start_v2 = np.nan
    stop_v2 = np.nan

    if somatic_mutation_sub_dict:  # exist maf dictionary, so we use germline mutation-applied seq as the background seq
        ref_seq = ref_mut_seq['background']
    else:
        ref_seq = ref_mut_seq['ref']
    mut_seq = ref_mut_seq['background']

    if strand == '+':
        coord = Coord(start_v1, stop_v1, start_v2, stop_v2)
        peptide_dna_str_mut = get_sub_mut_dna(mut_seq, coord, variant_comb, somatic_mutation_sub_dict, strand, gene_start)
        peptide_dna_str_ref = ref_seq[start_v1 - gene_start:stop_v1 - gene_start]
    else:
        coord = Coord(start_v1, stop_v1, start_v2, stop_v2)
        peptide_dna_str_mut = complementary_seq(get_sub_mut_dna(mut_seq, coord, variant_comb, somatic_mutation_sub_dict, strand, gene_start))
        peptide_dna_str_ref = complementary_seq(ref_seq[start_v1 - gene_start:stop_v1 - gene_start][::-1])

    peptide_mut, mut_has_stop_codon = dna_to_peptide(peptide_dna_str_mut, all_read_frames)
    peptide_ref, ref_has_stop_codon = dna_to_peptide(peptide_dna_str_ref, all_read_frames)

    is_isolated = True

    peptide = Peptide(peptide_mut,peptide_ref)
    coord = Coord(start_v1, stop_v1, start_v2, stop_v2)
    flag = Flag(mut_has_stop_codon,is_isolated)

    return peptide, coord, flag

def is_peptide_isolated(
    kmer: List[Tuple[int, int, int]],
    peptide_mut: List[str],
    segment_to_exons: Dict[int, Set[int]]) -> bool:
    """
    Determines whether the peptide translation is isolated to a single exon.

    Parameters
    ----------
    kmer : List of (seg_id, start, stop) tuples.
    peptide_mut : List of mutated peptide sequences (from dna_to_peptide).
    segment_to_exons : Mapping of segment IDs to sets of exon IDs.

    Returns
    -------
    bool: True if the peptide is isolated, False otherwise.
    """
    codon_len = len(peptide_mut[0]) * 3  # number of nucleotides translated

    translated_nt_count = 0
    translated_segment_ids = []

    for seg_id, seg_start, seg_end in kmer:
        seg_len = abs(seg_end - seg_start)
        if translated_nt_count >= codon_len:
            break # already have enough nucleotides
        translated_segment_ids.append(seg_id) # add segment id-s before the stop codon
        translated_nt_count += seg_len

    if len(translated_segment_ids) < 2:
        return True

    # Check if segment IDs are strictly consecutive (ascending or descending)
    diffs = [b - a for a, b in zip(translated_segment_ids, translated_segment_ids[1:])]
    is_consecutive = all(d == 1 for d in diffs) or all(d == -1 for d in diffs)

    if not is_consecutive: # if gaps, there has to be an intron
        return False

    # Check if all consecutive segments share at least one exon
    exon_sets = [segment_to_exons[sid] for sid in translated_segment_ids]
    common_exons = set.intersection(*exon_sets)
    return len(common_exons) > 0

def get_peptide_result(kmer: List[Tuple[int, int, int]],
                       strand: str,
                       variant_comb: List[int],
                       somatic_mutation_sub_dict: Dict[int, Dict[str, str]],
                       ref_mut_seq: Dict[str, str],
                       gene_start: int,
                       segment_to_exons: dict) -> Tuple[Kmer, Peptide, Flag]:
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
    segment_to_exons: Dict mapping seg_ids to exon_ids segment belong to
    otherwise a list of all translated peptides (for each stop codon) is provided.

    Returns
    -------
    Tuple of Kmer, Peptide and Flag objects.
    """

    # Choose correct background/reference sequences
    ref_seq = ref_mut_seq['background'] if somatic_mutation_sub_dict else ref_mut_seq['ref']
    mut_seq = ref_mut_seq['background']

    # Get sub-DNA strings (mutated and reference)
    dna_str_mut = get_sub_mut_dna(mut_seq, kmer, variant_comb, somatic_mutation_sub_dict, strand, gene_start)
    dna_str_ref = get_sub_mut_dna(ref_seq, kmer, np.nan, somatic_mutation_sub_dict, strand, gene_start)

    # Generate a complement for '-' strand (reverse done inside get_sub_mut_dna)
    if strand == "-":
        dna_str_mut = complementary_seq(dna_str_mut)
        dna_str_ref = complementary_seq(dna_str_ref)

    # Translate DNA to peptide
    peptide_mut, mut_has_stop_codon = dna_to_peptide(dna_str_mut, False)
    peptide_ref, ref_has_stop_codon = dna_to_peptide(dna_str_ref, False)

    # Check if the output peptide is translated from a single exon
    is_isolated = is_peptide_isolated(kmer, peptide_mut, segment_to_exons)
        
    peptide = Peptide(peptide_mut, peptide_ref)
    flag = Flag(mut_has_stop_codon, is_isolated)

    return peptide, flag


def get_exhaustive_reading_frames(splicegraph, gene_strand, vertex_order):

    # get the reading_frames
    reading_frames = {}
    for idx in vertex_order:
        reading_frames[idx] = set()
        v_start = splicegraph.vertices[0, idx]
        v_stop = splicegraph.vertices[1, idx]


        # Initialize reading regions from the CDS transcript annotations
        for cds_phase in [0, 1, 2]:
            if gene_strand== "-":
                cds_right_modi = max(v_stop - cds_phase, v_start)
                cds_left_modi = v_start
            else: #gene_strand == "+":
                cds_left_modi = min(v_start + cds_phase, v_stop)
                cds_right_modi = v_stop
            n_trailing_bases = cds_right_modi - cds_left_modi
            read_phase = n_trailing_bases % 3
            reading_frames[idx].add(ReadingFrameTuple(cds_left_modi, cds_right_modi, read_phase, False )) #no reading frame annotation status recorded 
    return reading_frames
