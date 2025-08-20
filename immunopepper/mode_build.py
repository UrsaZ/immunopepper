# Python libraries
""""""
# Core operation of ImmunoPepper. Traverse splicegraph and get kmer/peptide output

import glob
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
from immunopepper.io_ import collect_results, remove_folder_list

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
    global countinfo
    global genetable
    global kmer_database
    countinfo = countinfo_glob
    genetable = genetable_glob
    kmer_database = kmer_database_glob


def mapper_funct(tuple_arg):
    process_gene_batch_foreground(*tuple_arg)


def mapper_funct_back(tuple_arg):
    process_gene_batch_background(*tuple_arg)


def process_gene_batch_background(output_sample, mutation_sample, genes, gene_idxs, n_genes, mutation,
                                  genetable, arg, outbase, filepointer, verbose=False):
    if arg.parallel > 1:
        batch_name = int(outbase.split('/')[-1].split('_')[-1])
    else:
        batch_name = 'all'

    if (arg.parallel == 1) or (not os.path.exists(os.path.join(outbase, "Annot_IS_SUCCESS"))):
        pathlib.Path(outbase).mkdir(exist_ok=True, parents=True)
        set_pept_backgrd = set()
        set_kmer_backgrd = set()
        time_per_gene = []
        mem_per_gene = []
        all_gene_idxs = []

        for i, gene in enumerate(genes):
            # measure time
            start_time = timeit.default_timer()

            # Genes not contained in the annotation...
            if (gene.name not in genetable.gene_to_cds_begin or
                    gene.name not in genetable.gene_to_ts):
                continue

            all_gene_idxs.append(gene_idxs[i])

            chrm = gene.chr.strip()
            sub_mutation = get_sub_mutations(mutation, mutation_sample, chrm)
            ref_mut_seq = collect_background_transcripts(gene=gene, ref_seq_file=arg.ref_path,
                                                         chrm=chrm, mutation=sub_mutation)

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

        save_bg_peptide_set(set_pept_backgrd, filepointer, outbase, verbose)
        set_pept_backgrd.clear()
        save_bg_kmer_set(set_kmer_backgrd, filepointer, outbase, verbose)
        set_kmer_backgrd.clear()

        pathlib.Path(os.path.join(outbase, "Annot_IS_SUCCESS")).touch()

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
                                  genes_info, gene_idxs, n_genes, genes_interest, disable_process_libsize,
                                  all_read_frames, complexity_cap, mutation, junction_dict,
                                  arg, outbase, filepointer, verbose):
    global countinfo
    global genetable
    global kmer_database
    mut_count_id = None
    if arg.parallel > 1:
        batch_name = int(outbase.split('/')[-1].split('_')[-1])
    else:
        batch_name = 'all'

    if (arg.parallel == 1) or (not os.path.exists(os.path.join(outbase, "output_sample_IS_SUCCESS"))):
        pathlib.Path(outbase).mkdir(exist_ok=True, parents=True)
        set_pept_forgrd = set()
        time_per_gene = []
        mem_per_gene = []
        all_gene_idxs = []
        gene_expr = []

        for i, gene in enumerate(genes):
            # measure time
            start_time = timeit.default_timer()

            # Genes not contained in the annotation in annotated CDS mode
            if (gene.name not in genetable.gene_to_cds_begin or
                    gene.name not in genetable.gene_to_ts):
                continue

            idx = get_idx(countinfo, output_sample, gene_idxs[i])
            # Gene counts information
            # Gene of interest always compute expression, others compute expession if required for library
            if not disable_process_libsize or (gene.name in genes_interest):
                if countinfo:
                    gidx = countinfo.gene_idx_dict[gene.name]

                    with h5py.File(countinfo.h5fname, 'r') as h5f:
                        # Get edge counts
                        if not (gidx in countinfo.gene_id_to_edgerange and gidx in countinfo.gene_id_to_segrange):
                            edge_idxs = None
                            edge_counts = None
                        else:
                            edge_gene_idxs = list(np.arange(countinfo.gene_id_to_edgerange[gidx][0],
                                                            countinfo.gene_id_to_edgerange[gidx][1]))
                            edge_idxs = h5f['edge_idx'][edge_gene_idxs].astype('int')
                            edge_counts = h5f['edges'][edge_gene_idxs, :]  # will compute expression on whole graph

                        # Get segment counts
                        seg_gene_idxs = np.arange(countinfo.gene_id_to_segrange[gidx][0],
                                                  countinfo.gene_id_to_segrange[gidx][1])
                        seg_counts = h5f['segments'][seg_gene_idxs, :]
                        if output_samples_ids is not None:
                            seg_counts = seg_counts[:, output_samples_ids]  # limitation fancy hdf5 indexing
                        else:
                            output_samples_ids = np.arange(seg_counts.shape[1])
                else:
                    edge_idxs = None
                    edge_counts = None
                    seg_counts = None

            # library size calculated only for genes with CDS --> already checked in if-statement above
            if countinfo and not disable_process_libsize:
                gene_expr.append([gene.name] + get_total_gene_expr(gene, countinfo, seg_counts))

            # Process only gene quantification and library sizes
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
            sub_mutation = get_sub_mutations(mutation, mutation_sample, chrm)
            if arg.mutation_sample is not None:
                mut_count_id = [idx for idx, sample in enumerate(arg.output_samples)
                                if arg.mutation_sample.replace('-', '').replace('_', '').replace('.', '').replace('/', '') == sample]
            junction_list = None
            if junction_dict is not None and chrm in junction_dict:
                junction_list = junction_dict[chrm]

            pathlib.Path(get_save_path(filepointer.kmer_segm_expr_fp, outbase)).mkdir(exist_ok=True, parents=True)
            pathlib.Path(get_save_path(filepointer.kmer_edge_expr_fp, outbase)).mkdir(exist_ok=True, parents=True)
            pathlib.Path(get_save_path(filepointer.junction_meta_fp, outbase)).mkdir(exist_ok=True, parents=True)
            if arg.output_fasta:
                pathlib.Path(get_save_path(filepointer.junction_peptide_fp, outbase)).mkdir(exist_ok=True, parents=True)

            vertex_pairs, \
            ref_mut_seq, \
            exon_som_dict = collect_vertex_pairs(gene=gene,
                                                 gene_info=genes_info[i],
                                                 ref_seq_file=arg.ref_path,
                                                 chrm=chrm,
                                                 idx=idx,
                                                 mutation=sub_mutation,
                                                 all_read_frames=all_read_frames,
                                                 disable_concat=arg.disable_concat,
                                                 kmer_length=arg.kmer,
                                                 filter_redundant=arg.filter_redundant)

            get_and_write_peptide_and_kmer(peptide_set=set_pept_forgrd,
                                           gene=gene,
                                           all_vertex_pairs=vertex_pairs,
                                           ref_mut_seq=ref_mut_seq,
                                           idx=idx,
                                           exon_som_dict=exon_som_dict,
                                           countinfo=countinfo,
                                           mutation=sub_mutation,
                                           mut_count_id=mut_count_id,
                                           table=genetable,
                                           junction_list=junction_list,
                                           kmer_database=kmer_database,
                                           kmer=arg.kmer,
                                           force_ref_peptides=arg.force_ref_peptides,
                                           out_dir=outbase,
                                           edge_idxs=edge_idxs,
                                           edge_counts=edge_counts,
                                           seg_counts=seg_counts,
                                           all_read_frames=arg.all_read_frames,
                                           filepointer=filepointer,
                                           graph_output_samples_ids=output_samples_ids,
                                           graph_samples=arg.output_samples,
                                           verbose_save=verbose,
                                           fasta_save=arg.output_fasta)

            time_per_gene.append(timeit.default_timer() - start_time)
            mem_per_gene.append(print_memory_diags(disable_print=True))
            all_gene_idxs.append(gene_idxs[i])

        save_gene_expr_distr(gene_expr, arg.output_samples, output_sample,  filepointer, outbase, verbose)

        pathlib.Path(os.path.join(outbase, "output_sample_IS_SUCCESS")).touch()

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

def merge_parallel_results(output_path, mutation_mode, filepointer, arg):
    """Merge/reorganize parallel batch results to match non-parallel structure"""
    import glob
    import pandas as pd
    import os
    import shutil
    import gzip
    
    logging.info(">>>>>>>>> Merging parallel results")
    
    # Find all batch directories
    batch_dirs = glob.glob(os.path.join(output_path, f'tmp_out_{mutation_mode}_batch_*'))
    
    if not batch_dirs:
        logging.error("No batch directories found")
        return
    
    logging.info(f"Found {len(batch_dirs)} batch directories")
    
    # 1. Merge gene_expression_detail files and sort by gene column
    if countinfo:
        logging.info("Merging and sorting gene_expression_detail")
        gene_expr_files = []
        for batch_dir in batch_dirs:
            gene_expr_file = os.path.join(batch_dir, 'gene_expression_detail.gz')
            if os.path.isfile(gene_expr_file):
                gene_expr_files.append(gene_expr_file)
        
        if gene_expr_files:
            dfs = []
            for file_path in gene_expr_files:
                try:
                    df = pd.read_csv(file_path, sep='\t', compression='gzip')
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logging.warning(f"Failed to read {file_path}: {e}")
            
            if dfs:
                merged_df = pd.concat(dfs, ignore_index=True)
                merged_df = merged_df.sort_values(by=merged_df.columns[0], key=lambda x: pd.Categorical(x, categories=sorted(x.unique(), key=lambda s: int(s.replace('gene', '')))))
                
                final_path = os.path.join(output_path, 'gene_expression_detail')
                merged_df.to_csv(final_path, sep='\t', index=False)
                logging.info(f"Merged and sorted {len(merged_df)} gene expression records")
    
    # 2. Merge annotation files - FIXED: Added .gz extensions
    annotation_files = [
        f'{mutation_mode}_annot_kmer.gz',
        f'{mutation_mode}_annot_peptides.fa.gz'
    ]
    
    for filename in annotation_files:
        logging.info(f"Merging {filename}")
        all_files = []
        
        for batch_dir in batch_dirs:
            file_path = os.path.join(batch_dir, filename)
            if os.path.isfile(file_path):
                # Check if file has content
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > 0:
                        all_files.append(file_path)
                        logging.debug(f"Found non-empty file: {file_path} (size: {file_size} bytes)")
                    else:
                        logging.debug(f"Found empty file: {file_path}")
                except Exception as e:
                    logging.warning(f"Error checking file size for {file_path}: {e}")
        
        logging.info(f"Found {len(all_files)} non-empty files for {filename}")
        
        if all_files:
            # Remove .gz from final filename
            final_filename = filename.replace('.gz', '')
            final_path = os.path.join(output_path, final_filename)
            
            if filename.endswith('.fa.gz'):
                # Merge FASTA files - FIXED: Handle "fasta" header properly
                all_sequences = []
                for fasta_file in all_files:
                    try:
                        with gzip.open(fasta_file, 'rt') as f:
                            content = f.read().strip()
                            if content:
                                lines = content.split('\n')
                                # Skip the first line if it's just "fasta" header
                                start_idx = 1 if lines and lines[0].strip() == 'fasta' else 0
                                # Get all lines after the header (sequence IDs and sequences)
                                sequences = [line.strip() for line in lines[start_idx:] if line.strip()]
                                all_sequences.extend(sequences)
                                logging.debug(f"Read {len(sequences)} sequence lines from {fasta_file}")
                    except Exception as e:
                        logging.warning(f"Failed to read FASTA file {fasta_file}: {e}")
                
                if all_sequences:
                    with open(final_path, 'w') as f:
                        f.write("fasta\n")  # Write single header
                        f.write('\n'.join(all_sequences))  # Write all sequences
                    logging.info(f"Merged {len(all_sequences)} FASTA lines to {final_filename}")
                else:
                    logging.warning(f"No FASTA sequences found for {filename}")
                    # Create empty file with just header
                    with open(final_path, 'w') as f:
                        f.write("fasta\n")
                    
            else:
                # Handle annot_kmer files specially - they have header + single line with tab-separated kmers
                if filename.endswith('_annot_kmer.gz'):
                    all_kmers = set()  # Use set to avoid duplicates
                    
                    for data_file in all_files:
                        try:
                            with gzip.open(data_file, 'rt') as f:
                                lines = f.read().strip().split('\n')
                                if len(lines) >= 2:  # Header + data line
                                    data_line = lines[1]  # Second line contains the kmers
                                    kmers = data_line.split('\t')
                                    all_kmers.update(kmer.strip() for kmer in kmers if kmer.strip())
                                    logging.debug(f"Read {len(kmers)} kmers from {data_file}")
                        except Exception as e:
                            logging.warning(f"Failed to read annot_kmer file {data_file}: {e}")
                    
                    if all_kmers:
                        # Write merged kmers in the same format: header + tab-separated line
                        with open(final_path, 'w') as f:
                            f.write("kmer\n")
                            f.write("\t".join(sorted(all_kmers)))
                        logging.info(f"Merged {len(all_kmers)} unique kmers to {final_filename}")
                    else:
                        logging.warning(f"No valid kmers found for {filename}")
                        # Create empty file with just header
                        with open(final_path, 'w') as f:
                            f.write("kmer\n")
                else:
                    # Handle other TSV files normally
                    dfs = []
                    for data_file in all_files:
                        try:
                            df = pd.read_csv(data_file, sep='\t', compression='gzip')
                            if not df.empty:
                                dfs.append(df)
                                logging.debug(f"Read {len(df)} records from {data_file}")
                        except Exception as e:
                            logging.warning(f"Failed to read {data_file}: {e}")
                    
                    if dfs:
                        merged_df = pd.concat(dfs, ignore_index=True)
                        merged_df.to_csv(final_path, sep='\t', index=False)
                        logging.info(f"Merged {len(merged_df)} records to {final_filename}")
                    else:
                        logging.warning(f"No valid data found for {filename}")
                        # Create empty file
                        pd.DataFrame().to_csv(final_path, sep='\t', index=False)
            
        else:
            logging.warning(f"No non-empty files found for {filename} - creating empty file")
            final_filename = filename.replace('.gz', '')
            final_path = os.path.join(output_path, final_filename)
            if filename.endswith('.fa.gz'):
                # Create empty FASTA
                with open(final_path, 'w') as f:
                    pass
            else:
                # Create empty TSV
                pd.DataFrame().to_csv(final_path, sep='\t', index=False)
    
    # 3. Move partitioned directories (don't merge, just consolidate into single directories)
    partitioned_dirs = [
        f'{mutation_mode}_graph_kmer_JuncExpr',
        f'{mutation_mode}_graph_kmer_SegmExpr', 
        f'{mutation_mode}_sample_peptides_meta',
        f'{mutation_mode}_sample_peptides.fa'
    ]
    
    for dirname in partitioned_dirs:
        # Skip FASTA directory if not needed
        if dirname.endswith('.fa') and not arg.output_fasta:
            continue
            
        logging.info(f"Consolidating {dirname}")
        final_dir = os.path.join(output_path, dirname)
        os.makedirs(final_dir, exist_ok=True)
        
        # Collect all part files from all batch directories
        part_count = 0
        for batch_dir in batch_dirs:
            source_dir = os.path.join(batch_dir, dirname)
            if os.path.isdir(source_dir):
                # Move all part files from this batch directory (including .gz files)
                part_files = glob.glob(os.path.join(source_dir, 'part-*'))
                for part_file in part_files:
                    dest_file = os.path.join(final_dir, os.path.basename(part_file))
                    # Rename if file already exists (shouldn't happen but just in case)
                    counter = 1
                    while os.path.exists(dest_file):
                        name, ext = os.path.splitext(os.path.basename(part_file))
                        if ext == '.gz':
                            name_base, ext_base = os.path.splitext(name)
                            dest_file = os.path.join(final_dir, f"{name_base}_{counter}{ext_base}.gz")
                        else:
                            dest_file = os.path.join(final_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(part_file, dest_file)
                    part_count += 1
        
        if part_count > 0:
            logging.info(f"Moved {part_count} part files to {dirname}")
        else:
            logging.warning(f"No part files found for {dirname}")
    
    # 4. Copy success flags
    success_flags = ['output_sample_IS_SUCCESS', 'Annot_IS_SUCCESS']
    
    for flag_name in success_flags:
        # Find the flag in any batch directory and copy it to main output
        for batch_dir in batch_dirs:
            flag_path = os.path.join(batch_dir, flag_name)
            if os.path.exists(flag_path):
                final_flag_path = os.path.join(output_path, flag_name)
                if os.path.isfile(flag_path):
                    shutil.copy2(flag_path, final_flag_path)
                else:
                    # It's a directory or other type, just create empty file
                    open(final_flag_path, 'a').close()
                logging.info(f"Copied {flag_name} flag")
                break
    
    # Clean up batch directories
    logging.info("Cleaning up batch directories")
    for batch_dir in batch_dirs:
        try:
            shutil.rmtree(batch_dir)
        except Exception as e:
            logging.error(f"Failed to remove {batch_dir}: {e}")
    
    logging.info(">>>>>>>>> Parallel merging completed")

def mode_build(arg):
    global output_sample
    global filepointer
    global countinfo  # Will be used in non parallel mode
    global genetable  # Will be used in non parallel mode
    global kmer_database  # Will be used in non parallel mode

    # read and process the annotation file
    logging.info(">>>>>>>>> Build: Start Preprocessing")
    logging.info('Building lookup structure ...')
    start_time = timeit.default_timer()
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
    mutation = load_mutations(arg.germline, arg.somatic, arg.mutation_sample, arg.heter_code,
                              arg.pickle_samples if arg.use_mut_pickle else None,
                              arg.sample_name_map, arg.output_dir if arg.use_mut_pickle else None)

    # load splice graph
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

    check_chr_consistence(chromosome_set, mutation, graph_data)

    # read the intron of interest file gtex_junctions.hdf5
    junction_dict = parse_junction_meta_info(arg.gtex_junction_path)

    # read and process uniprot file
    kmer_database = parse_uniprot(arg.kmer_database)

    # process the genes according to the annotation file
    # add CDS starts and reading frames to the respective nodes
    logging.info('Add reading frame to splice graph ...')
    start_time = timeit.default_timer()
    graph_info = genes_preprocess_all(graph_data, genetable.gene_to_cds_begin,
                                      arg.parallel, arg.all_read_frames)
    end_time = timeit.default_timer()
    logging.info('\tTime spent: {:.3f} seconds'.format(end_time - start_time))
    print_memory_diags()
    logging.info(">>>>>>>>> Finish Preprocessing")

    # parse user choice for genes
    graph_data, genes_interest, n_genes, \
    complexity_cap, disable_process_libsize = parse_gene_choices(arg.genes_interest, arg.process_chr, arg.process_num,
                                                                 arg.complexity_cap, arg.disable_process_libsize,
                                                                 graph_data)

    # parse output_sample relatively to output mode
    process_output_samples, output_samples_ids = parse_output_samples_choices(arg, countinfo, matching_count_ids,
                                                                              matching_count_samples)

    logging.info(">>>>>>>>> Start traversing splice graph")
    for output_sample in process_output_samples:
        logging.info(f'>>>> Processing output_sample {output_sample}, there are {n_genes} graphs in total')

        # prepare the output files
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
                         graph_info[gene_idx], gene_idx, n_genes, genes_interest, disable_process_libsize,
                         arg.all_read_frames, complexity_cap, mutation, junction_dict, arg,
                         os.path.join(output_path, f'tmp_out_{mutation.mode}_batch_{i + arg.start_id}'),
                         filepointer, verbose_save) for i, gene_idx in gene_batches]
                result = pool.imap(mapper_funct, args, chunksize=1)
                exits_if_exception = [res for res in result]

            logging.info("Finished traversal")
                


        else:
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
                                          graph_info, genes_range, n_genes, genes_interest, disable_process_libsize,
                                          arg.all_read_frames, complexity_cap, mutation, junction_dict,
                                          arg, output_path, filepointer,
                                          verbose=True)

        if (not disable_process_libsize) and countinfo:
            output_libsize_fp = os.path.join(arg.output_dir, 'expression_counts.libsize.tsv')
            create_libsize(filepointer.gene_expr_fp, output_libsize_fp, output_path, mutation.mode, arg.parallel)

        if arg.parallel > 1:
            try:
                merge_parallel_results(output_path, mutation.mode, filepointer, arg)
            except Exception as e:
                logging.error(f"Failed to merge parallel results: {e}")
                raise
