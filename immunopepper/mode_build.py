# Python libraries
""""""
# Core operation of ImmunoPepper. Traverse splicegraph and get kmer/peptide output.
# Inputs:
#   - Splicegraph: a pickle file containing the splicegraph data
#   - Annotation: a gtf file containing the gene annotation
#   - Mutations: germline and somatic mutations in a pickle file
#   - Count data: a hdf5 file containing the gene expression counts (optional)
#   - Junctions: a hdf5 file containing the junction metadata (optional)
#   - Uniprot: a fasta file containing the k-mer database (optional)
# Outputs:
#   - Foreground k-mers/peptides: a set of k-mers/peptides generated from the splicegraph
#   - Background k-mers/peptides: a set of k-mers/peptides generated from the annotation
#   - Gene expression distribution: a tsv file containing the gene expression distribution
#   - Library size: a tsv file containing the library size for each sample
#   - Junction metadata: a hdf5 file containing the junction metadata
#   - Junction peptides: a fasta file containing the junction peptides (optional)

#   Foreground: novel peptides/k-mers from the splicegraph
#   Background: reference peptides/k-mers from the annotation (for filtering or comparison)

import glob
import pandas as pd
import shutil
import gzip
import h5py
import logging
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import pathlib
import pickle
import sys
import timeit

# immuno module
from immunopepper.io_ import get_save_path
from immunopepper.io_ import initialize_fp
from immunopepper.io_ import save_bg_kmer_set
from immunopepper.io_ import save_bg_peptide_set
from immunopepper.io_ import save_gene_expr_distr

from immunopepper.mutations import get_sub_mutations
from immunopepper.mutations import load_mutations
from immunopepper.preprocess import genes_preprocess_all
from immunopepper.preprocess import parse_junction_meta_info
from immunopepper.preprocess import parse_gene_choices
from immunopepper.preprocess import parse_gene_metadata_info
from immunopepper.preprocess import parse_output_samples_choices
from immunopepper.preprocess import parse_uniprot
from immunopepper.preprocess import preprocess_ann
from immunopepper.traversal import collect_background_transcripts
from immunopepper.traversal import collect_vertex_pairs
from immunopepper.traversal import get_and_write_background_peptide_and_kmer
from immunopepper.traversal import get_and_write_peptide_and_kmer
from immunopepper.utils import check_chr_consistence
from immunopepper.utils import create_libsize
from immunopepper.utils import get_idx
from immunopepper.utils import get_total_gene_expr
from immunopepper.utils import print_memory_diags
from immunopepper.traversal_optimised import get_kmers_and_peptides


# intermediate fix to load pickle files stored under previous version
from spladder.classes import gene as cgene
from spladder.classes import splicegraph as csplicegraph
from spladder.classes import segmentgraph as csegmentgraph
sys.modules['modules'] = cgene
sys.modules['modules.classes.gene'] = cgene
sys.modules['modules.classes.splicegraph'] = csplicegraph
sys.modules['modules.classes.segmentgraph'] = csegmentgraph
# end fix


def pool_initializer_glob(countinfo_glob, genetable_glob, kmer_database_glob):  # Moved from utils because of global variables
    """
    Initialize global variables for the multiprocessing pool.
    """
    global countinfo
    global genetable
    global kmer_database
    countinfo = countinfo_glob
    genetable = genetable_glob
    kmer_database = kmer_database_glob


def mapper_funct(tuple_arg):
    """
    Simple wrapper for unpacking arguments for the foreground gene batch processor inside multiprocessing.
    """
    process_gene_batch_foreground(*tuple_arg)


def mapper_funct_back(tuple_arg):
    """
    Simple wrapper for unpacking arguments for the background gene batch processor inside multiprocessing.
    """
    process_gene_batch_background(*tuple_arg)


def process_gene_batch_background(output_sample, mutation_sample, genes, gene_idxs, n_genes, mutation,
                                  genetable, arg, outbase, filepointer, verbose=False):
    """
    For each gene, generate reference k-mers/peptides by traversing its normal splice graph.
    """

    # Assign batch name from output folder name or "all" if single-threaded
    if arg.parallel > 1:
        batch_name = int(outbase.split('/')[-1].split('_')[-1])
    else:
        batch_name = 'all'

    # Avoid reprocessing: process the genes only if parallel mode is not used or the batch does not exist
    if (arg.parallel == 1) or (not os.path.exists(os.path.join(outbase, "Annot_IS_SUCCESS"))):
        pathlib.Path(outbase).mkdir(exist_ok=True, parents=True) # Create output directory if it does not exist
        # Initialize sets for background peptides and kmers, and lists for time and memory measurements
        set_pept_backgrd = set()
        set_kmer_backgrd = set()
        time_per_gene = []
        mem_per_gene = []
        all_gene_idxs = []

        # Loop over genes in batch
        for i, gene in enumerate(genes):
            # measure time
            start_time = timeit.default_timer()

            # Skip genes not contained in the annotation...
            if (gene.name not in genetable.gene_to_cds_begin or
                    gene.name not in genetable.gene_to_ts):
                continue

            all_gene_idxs.append(gene_idxs[i]) # Store gene index

            chrm = gene.chr.strip() # Get chromosome name
            sub_mutation = get_sub_mutations(mutation, mutation_sample, chrm) # Get sub-mutations for the chromosome
            # Collect the gene's background reference sequence with applied mutations.
            ref_mut_seq = collect_background_transcripts(gene=gene, ref_seq_file=arg.ref_path,
                                                         chrm=chrm, mutation=sub_mutation)

            # Generate and save background peptides and kmers for this gene to the sets
            get_and_write_background_peptide_and_kmer(peptide_set=set_pept_backgrd,
                                                      kmer_set=set_kmer_backgrd,
                                                      gene=gene,
                                                      ref_mut_seq=ref_mut_seq,
                                                      gene_table=genetable,
                                                      countinfo=None,
                                                      kmer_length=arg.kmer,
                                                      all_read_frames=arg.all_read_frames)

            time_per_gene.append(timeit.default_timer() - start_time)
            mem_per_gene.append(print_memory_diags(disable_print=True))

        # Save the background sets to disk, clear the sets after saving
        save_bg_peptide_set(set_pept_backgrd, filepointer, outbase, verbose)
        set_pept_backgrd.clear()
        save_bg_kmer_set(set_kmer_backgrd, filepointer, outbase, verbose)
        set_kmer_backgrd.clear()

        pathlib.Path(os.path.join(outbase, "Annot_IS_SUCCESS")).touch() # Create a file to indicate successful processing

        if time_per_gene:
            logging_string = (f'....{output_sample}: annotation graph from batch {batch_name}/{n_genes} '
                              f'processed, max time cost: {np.round(np.nanmax(time_per_gene), 2)}, '
                              f'memory cost: {np.round(np.nanmax(mem_per_gene), 2)} GB')
            logging.debug(logging_string)
        else:
            logging_string = f'....{output_sample}: output_sample graph from batch {batch_name}/{n_genes}, no processing'

        if (batch_name != 'all') and (batch_name % 10000 == 0):
            logging.info(logging_string)

    else:  # Batch has already been saved to disk
        logging_string = f'> {output_sample} : Batch {batch_name} exists, skip processing'
        logging.debug(logging_string)
        if (batch_name != 'all') and (batch_name % 10000 == 0):
            logging.info(logging_string)

    return 'multiprocessing is success'


def process_gene_batch_foreground(output_sample, mutation_sample, output_samples_ids, genes,
                                  gene_idxs, n_genes, genes_interest, disable_process_libsize,
                                  all_read_frames, complexity_cap, mutation, junction_dict,
                                  arg, outbase, filepointer, verbose):
    """
    For each gene, generate k-mers/peptides by traversing its splice graph.
    Apply sample-specific mutations, expression data, junctions.
    """
    global countinfo
    global genetable
    global kmer_database
    mut_count_id = None

    # Assign batch name from output folder name or "all" if single-threaded
    if arg.parallel > 1:
        batch_name = int(outbase.split('/')[-1].split('_')[-1])
    else:
        batch_name = 'all'

    # Avoid reprocessing: process the genes only if parallel mode is not used or the batch does not exist
    if (arg.parallel == 1) or (not os.path.exists(os.path.join(outbase, "output_sample_IS_SUCCESS"))):
        pathlib.Path(outbase).mkdir(exist_ok=True, parents=True) # Create output directory if it does not exist
        # Initialize sets for foreground peptides and kmers, and lists for time and memory measurements
        set_pept_forgrd = set()
        time_per_gene = []
        mem_per_gene = []
        all_gene_idxs = []
        gene_expr = []

        # Loop over genes in batch
        for i, gene in enumerate(genes):
            start_time = timeit.default_timer() # measure time

            # Skip genes not contained in the annotation in annotated CDS mode
            if (gene.name not in genetable.gene_to_cds_begin or
                    gene.name not in genetable.gene_to_ts):
                continue

            idx = get_idx(countinfo, output_sample, gene_idxs[i]) # Get index of the gene in the countinfo
            # Gene counts information
            # Gene of interest always compute expression, others compute expession if required for library
            if not disable_process_libsize or (gene.name in genes_interest):
                if countinfo:
                    gidx = countinfo.gene_idx_dict[gene.name] # Get gene index in the countinfo

                    with h5py.File(countinfo.h5fname, 'r') as h5f:
                        if not (gidx in countinfo.gene_id_to_edgerange and gidx in countinfo.gene_id_to_segrange):
                            edge_idxs = None
                            edge_counts = None
                        else: # Extract edge indices and counts if available.
                            edge_gene_idxs = list(np.arange(countinfo.gene_id_to_edgerange[gidx][0],
                                                            countinfo.gene_id_to_edgerange[gidx][1]))
                            edge_idxs = h5f['edge_idx'][edge_gene_idxs].astype('int')
                            edge_counts = h5f['edges'][edge_gene_idxs, :]  # will compute expression on whole graph

                        # Get segment counts
                        seg_gene_idxs = np.arange(countinfo.gene_id_to_segrange[gidx][0],
                                                  countinfo.gene_id_to_segrange[gidx][1])
                        seg_counts = h5f['segments'][seg_gene_idxs, :]
                        if output_samples_ids is not None:  # If output_samples_ids is provided, filter segment counts
                            seg_counts = seg_counts[:, output_samples_ids]  # limitation fancy hdf5 indexing
                        else:
                            output_samples_ids = np.arange(seg_counts.shape[1]) 
                else: # If countinfo is not available, set edge and segment counts to None
                    edge_idxs = None
                    edge_counts = None
                    seg_counts = None

            # library size calculated only for genes with CDS --> already checked in if-statement above
            if countinfo and not disable_process_libsize:
                # Get total gene expression for the gene as:
                # total_expr = reads_length*total_reads_counts
                gene_expr.append([gene.name] + get_total_gene_expr(gene, countinfo, seg_counts))

            # Process only gene quantification and library sizes, skip further processing
            if arg.libsize_extract:
                time_per_gene.append(timeit.default_timer() - start_time)
                mem_per_gene.append(print_memory_diags(disable_print=True))
                continue

            # Process only gene of interest
            if gene.name not in genes_interest:
                continue

            # Do not process genes with highly complex splice graphs
            if len(gene.splicegraph.vertices[1]) > complexity_cap:
                logging.warning(f'> Gene {gene.name} has a edge complexity > {complexity_cap}, not processed')
                continue

            chrm = gene.chr.strip()
            # Get germline and somatic mutations for the given sample and chromosome
            sub_mutation = get_sub_mutations(mutation, mutation_sample, chrm)
            # Collect the gene's background reference sequence with applied germline mutations.
            ref_mut_seq = collect_background_transcripts(gene=gene, ref_seq_file=arg.ref_path,
                                                         chrm=chrm, mutation=sub_mutation)
            if arg.mutation_sample is not None:
                mut_count_id = [idx for idx, sample in enumerate(arg.output_samples)
                                if arg.mutation_sample.replace('-', '').replace('_', '').replace('.', '').replace('/', '') == sample]
            # Extract relevant junction data if provided
            junction_list = None
            if junction_dict is not None and chrm in junction_dict:
                junction_list = junction_dict[chrm]

            # Prepare output directories 
            pathlib.Path(get_save_path(filepointer.kmer_segm_expr_fp, outbase)).mkdir(exist_ok=True, parents=True)
            pathlib.Path(get_save_path(filepointer.kmer_edge_expr_fp, outbase)).mkdir(exist_ok=True, parents=True)
            pathlib.Path(get_save_path(filepointer.junction_meta_fp, outbase)).mkdir(exist_ok=True, parents=True)
            if arg.output_fasta:
                pathlib.Path(get_save_path(filepointer.junction_peptide_fp, outbase)).mkdir(exist_ok=True, parents=True)

            get_kmers_and_peptides(gene=gene,
                                    mutation=sub_mutation,
                                    table=genetable,
                                    ref_seq_file=arg.ref_path,
                                    chrm=chrm,
                                    peptide_set=set_pept_forgrd,
                                    kmer_length=27,
                                    pep_length=9,
                                    idx=idx,
                                    countinfo=countinfo,
                                    edge_idxs=edge_idxs,
                                    edge_counts=edge_counts,
                                    seg_counts=seg_counts,
                                    mut_count_id=mut_count_id,
                                    junction_list=junction_list,
                                    kmer_database=kmer_database,
                                    filepointer=filepointer,
                                    force_ref_peptides=arg.force_ref_peptides,
                                    graph_output_samples_ids=output_samples_ids,
                                    graph_samples=arg.output_samples,
                                    out_dir=outbase,
                                    verbose_save=verbose,
                                    fasta_save=arg.output_fasta)                

            time_per_gene.append(timeit.default_timer() - start_time)
            mem_per_gene.append(print_memory_diags(disable_print=True))
            all_gene_idxs.append(gene_idxs[i])

        # Save gene expression data to a file
        save_gene_expr_distr(gene_expr, arg.output_samples, output_sample,  filepointer, outbase, verbose)

        pathlib.Path(os.path.join(outbase, "output_sample_IS_SUCCESS")).touch() # Create a file to indicate successful processing

        if time_per_gene:
            logging_string = (f'....{output_sample}: output_sample graph from batch {batch_name}/{n_genes} processed, '
                              f'max time cost: {np.round(np.nanmax(time_per_gene), 2)}, '
                              f'memory cost: {np.round(np.nanmax(mem_per_gene), 2)} GB')
            logging.debug(logging_string)
        else:
            logging_string = f'....{output_sample}: output_sample graph from batch {batch_name}/{n_genes}, no processing'
        if (batch_name != 'all') and (batch_name % 10000 == 0):
            logging.info(logging_string)

    else:  # Batch has already been saved to disk
        logging_string = f'> {output_sample}: Batch {batch_name} exists, skip processing'
        logging.debug(logging_string)
        if (batch_name != 'all') and (batch_name % 10000 == 0):
            logging.info(logging_string)

    return 'multiprocessing is success'


def merge_parallel_results(output_path, mutation_mode, arg):
    """Merge/reorganize parallel batch results to match non-parallel structure"""

    logging.info(">>>>>>>>> Merging parallel results")
    
    # Find all batch directories
    batch_dirs = glob.glob(os.path.join(output_path, f'tmp_out_{mutation_mode}_batch_*'))
    
    if not batch_dirs:
        logging.error("No batch directories found")
        return
    
    logging.info(f"Found {len(batch_dirs)} batch directories")
    
    # 1. Merge gene_expression_detail files if countinfo exists
    if countinfo:
        gene_expr_files = []
        for batch_dir in batch_dirs:
            gene_expr_file = os.path.join(batch_dir, 'gene_expression_detail.gz')
            if os.path.isfile(gene_expr_file) and os.path.getsize(gene_expr_file) > 0:
                gene_expr_files.append(gene_expr_file)
        
        if gene_expr_files:
            try:
                dfs = []
                for file_path in gene_expr_files:
                    df = pd.read_csv(file_path, sep='\t', compression='gzip')
                    if not df.empty:
                        dfs.append(df)
                
                if dfs:
                    merged_df = pd.concat(dfs, ignore_index=True)
                    # Sort by first column (gene names)
                    try:
                        merged_df = merged_df.sort_values(by=merged_df.columns[0], 
                                                        key=lambda x: pd.Categorical(x, categories=sorted(x.unique(), 
                                                        key=lambda s: int(s.replace('gene', '')) if 'gene' in s else 0)))
                    except:
                        # Fallback to simple sort if the above fails
                        merged_df = merged_df.sort_values(by=merged_df.columns[0])
                    
                    final_path = os.path.join(output_path, 'gene_expression_detail')
                    merged_df.to_csv(final_path, sep='\t', index=False)
                    logging.info(f"Merged {len(merged_df)} gene expression records")
            except Exception as e:
                logging.warning(f"Failed to merge gene expression files: {e}")
    
    # 2. Merge annotation files by content type
    annotation_patterns = {
        f'{mutation_mode}_annot_kmer.gz': 'kmer',
        f'{mutation_mode}_annot_peptides.fa.gz': 'fasta'
    }
    
    for filename, file_type in annotation_patterns.items():
        all_files = []
        for batch_dir in batch_dirs:
            file_path = os.path.join(batch_dir, filename)
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                all_files.append(file_path)
        
        final_filename = filename.replace('.gz', '')
        final_path = os.path.join(output_path, final_filename)
        
        if all_files:
            try:
                if file_type == 'fasta':
                    # Merge FASTA files
                    all_sequences = []
                    for fasta_file in all_files:
                        with gzip.open(fasta_file, 'rt') as f:
                            content = f.read().strip()
                            if content:
                                lines = content.split('\n')
                                # Skip header if it's just "fasta"
                                start_idx = 1 if lines and lines[0].strip() == 'fasta' else 0
                                sequences = [line.strip() for line in lines[start_idx:] if line.strip()]
                                all_sequences.extend(sequences)
                    
                    with open(final_path, 'w') as f:
                        if all_sequences:
                            f.write("fasta\n")
                            f.write('\n'.join(all_sequences))
                            logging.info(f"Merged {len(all_sequences)} FASTA background peptides")
                        else:
                            logging.info("No FASTA background peptide sequences found to merge")
                    
                elif file_type == 'kmer':
                    # Merge kmer files (header + tab-separated kmers)
                    all_kmers = set()
                    for kmer_file in all_files:
                        with gzip.open(kmer_file, 'rt') as f:
                            lines = f.read().strip().split('\n')
                            if len(lines) >= 2:
                                kmers = lines[1].split('\t')
                                all_kmers.update(k.strip() for k in kmers if k.strip())
                    
                    with open(final_path, 'w') as f:
                        f.write("kmer\n")
                        if all_kmers:
                            f.write("\t".join(sorted(all_kmers)))
                            logging.info(f"Merged {len(all_kmers)} background kmers")
                        else:
                            logging.info("No background kmers found to merge")
                    
            except Exception as e:
                logging.warning(f"Failed to merge {filename}: {e}")

    
    # 3. Move ALL subdirectories and their contents (simplified approach)
    # Collect all unique subdirectory names across all batches
    all_subdirs = set()
    for batch_dir in batch_dirs:
        if os.path.isdir(batch_dir):
            for item in os.listdir(batch_dir):
                item_path = os.path.join(batch_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    all_subdirs.add(item)
    
    # Move contents of each subdirectory type
    for subdir_name in all_subdirs:
        # Skip FASTA directories if not needed
        if subdir_name.endswith('.fa') and not arg.output_fasta:
            continue
            
        final_dir = os.path.join(output_path, subdir_name)
        os.makedirs(final_dir, exist_ok=True)
        
        total_files_moved = 0
        for batch_dir in batch_dirs:
            source_dir = os.path.join(batch_dir, subdir_name)
            if os.path.isdir(source_dir):
                # Move ALL files from this subdirectory
                for filename in os.listdir(source_dir):
                    source_file = os.path.join(source_dir, filename)
                    if os.path.isfile(source_file):
                        dest_file = os.path.join(final_dir, filename)
                        
                        # Handle filename conflicts by adding batch suffix
                        if os.path.exists(dest_file):
                            batch_id = batch_dir.split('_')[-1]  # Extract batch number
                            name, ext = os.path.splitext(filename)
                            dest_file = os.path.join(final_dir, f"{name}_batch{batch_id}{ext}")
                        
                        shutil.move(source_file, dest_file)
                        total_files_moved += 1
        
        logging.info(f"Moved {total_files_moved} files to {subdir_name}")
    
    # 4. Copy success flags
    success_flags = ['output_sample_IS_SUCCESS', 'Annot_IS_SUCCESS']
    for flag_name in success_flags:
        for batch_dir in batch_dirs:
            flag_path = os.path.join(batch_dir, flag_name)
            if os.path.exists(flag_path):
                final_flag_path = os.path.join(output_path, flag_name)
                if os.path.isfile(flag_path):
                    shutil.copy2(flag_path, final_flag_path)
                else:
                    # Create empty flag file
                    pathlib.Path(final_flag_path).touch()
                break
    
    # 5. Clean up batch directories
    logging.info("Cleaning up batch directories")
    for batch_dir in batch_dirs:
        try:
            shutil.rmtree(batch_dir)
            logging.debug(f"Removed {batch_dir}")
        except Exception as e:
            logging.error(f"Failed to remove {batch_dir}: {e}")
    
    logging.info(">>>>>>>>> Parallel merging completed")


def mode_build(arg): # main, handles setup, loading, and dispatc
    global output_sample
    global filepointer
    global countinfo  # Will be used in non parallel mode
    global genetable  # Will be used in non parallel mode
    global kmer_database  # Will be used in non parallel mode

    # read and process the annotation file
    logging.info(">>>>>>>>> Build: Start Preprocessing")
    logging.info('Building lookup structure ...')
    start_time = timeit.default_timer()

    # extract info from the annotation file:
    #   - gene table: stores the gene-transcript-cds mapping tables 
    #     ['gene_to_cds_begin', 'ts_to_cds', 'gene_to_cds']
    #   - chromosome set: stores chromosome names
    genetable, chromosome_set = preprocess_ann(arg.ann_path)
    end_time = timeit.default_timer()
    logging.info('\tTime spent: {:.3f} seconds'.format(end_time - start_time))
    print_memory_diags()

    # load graph metadata
    start_time = timeit.default_timer()
    if arg.count_path is not None:
        logging.info('Loading count data ...')
        countinfo, matching_count_samples, matching_count_ids = parse_gene_metadata_info(arg.count_path,
                                                                                         arg.output_samples)

        end_time = timeit.default_timer()
        logging.info('\tTime spent: {:.3f} seconds'.format(end_time - start_time))
        print_memory_diags()
    else:
        countinfo = None
        matching_count_samples = None
        matching_count_ids = None

    # read the variant file
    # mutation is a namedtuple('Mutation', ['mode', 'germline_dict', 'somatic_dict'])
    # dict:  Key is mutation position, value is mutations:: {10 : {'mut_base': '*', 'ref_base': 'A'}} 
    mutation = load_mutations(arg.germline, arg.somatic, arg.mutation_sample, arg.heter_code,
                              arg.pickle_samples if arg.use_mut_pickle else None,
                              arg.sample_name_map, arg.output_dir if arg.use_mut_pickle else None)

    # load splice graph
    # graph_data is an array with spladder.classes.gene.Gene objects
    # graph_meta is a dictionary
    logging.info('Loading splice graph ...')
    start_time = timeit.default_timer()
    with open(arg.splice_path, 'rb') as graph_fp:
        (graph_data, graph_meta) = pickle.load(graph_fp, encoding='latin1')  # both graph data and meta data
    end_time = timeit.default_timer()
    logging.info('\tTime spent: {:.3f} seconds'.format(end_time - start_time))
    print_memory_diags()

    # DEBUG
    # graph_data = graph_data[[3170]] #TODO remove
    # graph_data = graph_data[940:942]
    if arg.start_id != 0 and arg.start_id < len(graph_data):
        logging.info(f'development feature: starting at gene number {arg.start_id}')
        graph_data = graph_data[arg.start_id:]
    # Verify chromosome naming consistency between annotation, mutations, and splice graphs.
    check_chr_consistence(chromosome_set, mutation, graph_data)

    # read the intron of interest file gtex_junctions.hdf5
    junction_dict = parse_junction_meta_info(arg.gtex_junction_path)

    # read and process uniprot file
    kmer_database = parse_uniprot(arg.kmer_database)

    # process the genes according to the annotation file
    # add CDS starts and reading frames to the respective nodes
    logging.info('Add reading frame to splice graph ...')
    start_time = timeit.default_timer()
    #graph_info is a list with GeneInfo objects for each gene.
    # GeneInfo contains vertex_succ_list, vertex_order, ReadingFrameTuple(s), vertex_len_dict, nvertices
    graph_info = genes_preprocess_all(graph_data, genetable.gene_to_cds_begin,
                                      arg.parallel, arg.all_read_frames) #TODO: remove?
    end_time = timeit.default_timer()
    logging.info('\tTime spent: {:.3f} seconds'.format(end_time - start_time))
    print_memory_diags()
    logging.info(">>>>>>>>> Finish Preprocessing")

    # parse user choice for genes (filter)
    graph_data, genes_interest, n_genes, \
    complexity_cap, disable_process_libsize = parse_gene_choices(arg.genes_interest, arg.process_chr, arg.process_num,
                                                                 arg.complexity_cap, arg.disable_process_libsize,
                                                                 graph_data)

    # parse output_sample relatively to output mode: if no samples specified in arg, take all present in the countfile
    process_output_samples, output_samples_ids = parse_output_samples_choices(arg, countinfo, matching_count_ids,
                                                                              matching_count_samples)

    logging.info(">>>>>>>>> Start traversing splice graph")
    # Loop over output samples
    for output_sample in process_output_samples:
        logging.info(f'>>>> Processing output_sample {output_sample}, there are {n_genes} graphs in total')

        # Determine output directory name for each sample.
        if output_sample != arg.mutation_sample:
            output_path = os.path.join(arg.output_dir, f'{output_sample}_mut{arg.mutation_sample}')
        else:
            output_path = os.path.join(arg.output_dir, output_sample)
        logging.info(f'Saving results to {output_path}')

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        filepointer = initialize_fp(output_path, mutation.mode, arg.output_fasta)

        # go over each gene in splicegraph
        genes_range = list(range(0, n_genes))

        # Split genes into batches for multiprocessing.
        # Run background and foreground processing in parallel if requested.
        if arg.parallel > 1:
            logging.info(f'Parallel: {arg.parallel} Threads')
            batch_size = min(n_genes, arg.batch_size)
            verbose_save = False
            gene_batches = [(i, genes_range[i:min(i + batch_size, n_genes)]) for i in
                            range(0, n_genes, batch_size)]

            # Build the background if requested
            if (not arg.skip_annotation) and not arg.libsize_extract:
                logging.info(">>>>>>>>> Start Background processing")
                with ThreadPool(processes=arg.parallel, initializer=pool_initializer_glob, initargs=(countinfo, genetable, kmer_database)) as pool:
                    args = [(output_sample, arg.mutation_sample,  graph_data[gene_idx], gene_idx, n_genes, mutation,
                             genetable, arg,
                             os.path.join(output_path, f'tmp_out_{mutation.mode}_batch_{i + arg.start_id}'),
                             filepointer, verbose_save) for i, gene_idx in gene_batches]
                    result = pool.imap(mapper_funct_back, args, chunksize=1)
                    exits_if_exception = [res for res in result]

            # Build the foreground
            logging.info(">>>>>>>>> Start Foreground processing")
            with Pool(processes=arg.parallel, initializer=pool_initializer_glob, initargs=(countinfo, genetable, kmer_database)) as pool:
                args = [(output_sample, arg.mutation_sample, output_samples_ids, graph_data[gene_idx],
                         gene_idx, n_genes, genes_interest, disable_process_libsize,
                         arg.all_read_frames, complexity_cap, mutation, junction_dict, arg,
                         os.path.join(output_path, f'tmp_out_{mutation.mode}_batch_{i + arg.start_id}'),
                         filepointer, verbose_save) for i, gene_idx in gene_batches]
                result = pool.imap(mapper_funct, args, chunksize=1)
                exits_if_exception = [res for res in result]

            logging.info("Finished traversal")

        else: # Not parallel
            logging.info('Not Parallel')
            # Build the background
            if (not arg.skip_annotation) and not arg.libsize_extract:
                logging.info(">>>>>>>>> Start Background processing")
                process_gene_batch_background(output_sample, arg.mutation_sample, graph_data, genes_range, n_genes,
                                              mutation, genetable, arg, output_path, filepointer,
                                              verbose=True)
            # Build the foreground and remove the background if needed
            logging.info(">>>>>>>>> Start Foreground processing")
            process_gene_batch_foreground(output_sample, arg.mutation_sample, output_samples_ids, graph_data,
                                          genes_range, n_genes, genes_interest, disable_process_libsize,
                                          arg.all_read_frames, complexity_cap, mutation, junction_dict,
                                          arg, output_path, filepointer,
                                          verbose=True)

        if (not disable_process_libsize) and countinfo:
            output_libsize_fp = os.path.join(arg.output_dir, 'expression_counts.libsize.tsv')
            create_libsize(filepointer.gene_expr_fp, output_libsize_fp, output_path, mutation.mode, arg.parallel)
        
        if arg.parallel > 1:
            try:
                merge_parallel_results(output_path, mutation.mode, arg)
            except Exception as e:
                logging.error(f"Failed to merge parallel results: {e}")
                raise
