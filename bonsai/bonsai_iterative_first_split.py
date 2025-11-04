from argparse import ArgumentParser
import numpy as np
import time
from pathlib import Path
import os, sys, csv
import subprocess

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory of this script-file to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

from bonsai.bonsai_helpers import Run_Configs, remove_tree_folders, find_latest_tree_folder_new, \
    convert_dict_to_named_tuple, str2bool

parser = ArgumentParser(
    description='First script to be called in bonsai_iterative_build.py.'
                'Creates a first star-tree on a subset of the data. Preprocesses the rest of the data and to make it'
                'ready for being added in other scripts.')

parser.add_argument('--config_filepath', type=str, default=None,
                    help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
                         'needed to run Bonsai.')

parser.add_argument('--starting_cell_ids', type=str, default=None,
                    help="Filename where list of cell-IDs is stored on which we will build the first guide tree.")

args = parser.parse_args()

# TODO: Make sure that Run_configs initialization just copies all arguments that were originally in args
config_filepath = args.config_filepath
starting_cell_ids = args.starting_cell_ids

args = Run_Configs(args.config_filepath)

args.starting_cell_ids = starting_cell_ids
args.config_filepath = config_filepath

import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_dataprocessing import initializeSCData, getMetadata, loadReconstructedTreeAndData, SCData, \
    nnnReorder, nnnReorderRandom, Metadata, storeData
from bonsai.bonsai_helpers import mp_print, startMPI, getOutputFolder, get_latest_intermediate, \
    clean_up_redundant_data_files, str2bool
from bonsai.bonsai_treeHelpers import Tree
import bonsai.bonsai_globals as bs_glob

# Some of the code can be run in parallel, parallelization is done via mpi4py
mpi_rank, mpi_size = startMPI(verbose=True)

scdata = initializeSCData(args, createStarTree=False, getOrigData=True, otherRanksMinimalInfo=True)

cell_ids_subset = []
with open(args.starting_cell_ids, 'r') as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        cell_ids_subset.append(row[0])

orig_cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(scdata.metadata.cellIds)}
subset_inds = np.array([orig_cell_id_to_ind[cell_id] for cell_id in cell_ids_subset])

metadata_subset = Metadata(curr_metadata=scdata.metadata)
metadata_subset.cellIds = cell_ids_subset
metadata_subset.nCells = len(cell_ids_subset)
bs_glob.nCells = metadata_subset.nCells
metadata_subset.dataset = os.path.join(scdata.metadata.dataset, 'subset_0')
metadata_subset.results_folder = os.path.join(scdata.metadata.results_folder, 'subset_0')

scdata_subset = SCData(onlyObject=True, dataset=metadata_subset.dataset,
                       results_folder=metadata_subset.results_folder)

scdata_subset.metadata = metadata_subset
ltqs_subset = scdata.originalData.ltqs[:, subset_inds].copy()
ltqsvars_subset = scdata.originalData.ltqsVars[:, subset_inds].copy()

scdata_subset.metadata.processedDatafolder = scdata_subset.result_path('zscorefiltered_%.3f_and_processed' % args.zscore_cutoff)
storeData(scdata_subset.metadata, ltqs_subset, ltqsvars_subset)
mp_print("Storing subset {} with {} cells in folder: {}".format(0, len(subset_inds),
                                                                scdata_subset.metadata.processedDatafolder))
# storeData(scdata_subset.metadata, ltqs_subset, ltqsvars_subset)

if mpi_rank == 0:
    scdata_subset.tree = Tree()
    scdata_subset.tree.initialize_star_tree(ltqs_subset, ltqsvars_subset, scdata_subset.metadata,
                                            opt_times=True, verbose=args.verbose)
    # Store tree topology with optimised times, and the data only for selected genes, such that it can be read
    # in by multiple cores such that the next part of the program can be run in parallel
    outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, greedy=False,
                                   redo_starry=False, opt_times=False)
    mp_print("Storing result of preprocessing in " + scdata_subset.result_path(outputFolder) + "\n\n")
    scdata_subset.storeTreeInFolder(scdata_subset.result_path(outputFolder), with_coords=True, verbose=args.verbose,
                                    cleanup_tree=False)
