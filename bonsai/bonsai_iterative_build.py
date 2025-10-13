from argparse import ArgumentParser
import numpy as np
import time
from pathlib import Path
import os, sys, csv

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory of this script-file to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

from bonsai.bonsai_helpers import Run_Configs, remove_tree_folders, find_latest_tree_folder_new, \
    convert_dict_to_named_tuple, str2bool

parser = ArgumentParser(
    description='Starts from cell-data and a tree on which some of the cells are already placed. Cells that are not on'
                'the tree but are in the data will be placed one by one on the tree.')

parser.add_argument('--config_filepath', type=str, default=None,
                    help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
                         'needed to run Bonsai.')

# TODO: To be moved to config-file
parser.add_argument('--select_target', type=str, default='cluster_centers',
                    help="This will determine what strategy we follow for the "
                         "spr-moves. Current options are 'root', 'cluster_centers', 'exhaustive'")

parser.add_argument('--iterative_cell_lists', type=str, default=None,
                    help="[NOT YET IMPLEMENTED]"
                         "One can give a filename here for a file where we have all cell-IDs (one per line) in the "
                         "first column. Second column contains an integer indicating at which iteration the cells "
                         "should be added. Useful for modeling time-course data.")

parser.add_argument('--n_initial_cells', type=int, default=10000,
                    help="If --iterative_cell_lists is None, this number will determine the size of the subset of "
                         "cells on which we will make the initial guide-tree.")

parser.add_argument('--return_commands', type=str2bool, default=False,
                    help="If True, this script will only return a file with the commands that should be run, "
                         "but not run them. This is useful for running these commands using parallel computing.\n"
                         "If False, this script will just run these commands on a single CPU. MPI will not be used.")

parser.add_argument('--pickup_intermediate', type=str2bool, default=False,
                    help='Optional: Set this argument to true if Bonsai needs to search in the indicated results-folder'
                         'for the furthest developed tree reconstruction, and pick it up from there. If this argument'
                         'is True, it over-rules the same argument in bonsai_config.yaml, and the other way around.')

parser.add_argument('--seed', type=int, default=1231,
                    help="Sets the random seed necessary for getting the subset of cells.")

args = parser.parse_args()
pickup_intermediate = args.pickup_intermediate
select_target = args.select_target
iterative_cell_lists = args.iterative_cell_lists
n_initial_cells = args.n_initial_cells
seed = args.seed

args = Run_Configs(args.config_filepath)
args.select_target = select_target
args.iterative_cell_lists = iterative_cell_lists
args.n_initial_cells = n_initial_cells
if pickup_intermediate:
    args.pickup_intermediate = True

import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_dataprocessing import initializeSCData, getMetadata, loadReconstructedTreeAndData, SCData, \
    nnnReorder, nnnReorderRandom, Metadata, storeData
from bonsai.bonsai_helpers import mp_print, startMPI, getOutputFolder, get_latest_intermediate, \
    clean_up_redundant_data_files, str2bool
import bonsai.bonsai_globals as bs_glob

# Some of the code can be run in parallel, parallelization is done via mpi4py
mpi_rank, mpi_size = startMPI(verbose=True)

start_all = time.time()

"""
------------------------------------------------------------------------------------------
Step 1: Get a subset of the cells and store their information in a subfolder.
------------------------------------------------------------------------------------------
"""

scdata = initializeSCData(args, createStarTree=False, getOrigData=True, otherRanksMinimalInfo=True)

subsets = None
if args.iterative_cell_lists is not None:
    # TODO: Implement this still. Create list of np-arrays of cell-indices in subsets-variable.
    # Last set of cell-IDs should not be in there, is assumed to be the complement
    pass

if subsets is None:
    n_initial_cells = min(args.n_initial_cells, scdata.metadata.nCells)
    np.random.seed(seed)
    subsets = [np.random.choice(np.arange(scdata.metadata.nCells), size=n_initial_cells, replace=False)]

for subset_ind, subset in enumerate(subsets):
    metadata_subset = Metadata(curr_metadata=scdata.metadata)
    metadata_subset.cellIds = [scdata.metadata.cellIds[ind] for ind in subset]
    metadata_subset.nCells = len(subset)
    metadata_subset.dataset = os.path.join(scdata.metadata.dataset, 'subset_{}'.format(subset_ind))
    metadata_subset.results_folder = os.path.join(scdata.metadata.results_folder, 'subset_{}'.format(subset_ind))

    scdata_subset = SCData(onlyObject=True, dataset=metadata_subset.dataset,
                           results_folder=metadata_subset.results_folder)

    metadata_subset.processedDatafolder = scdata_subset.result_path('zscorefiltered_%.3f_and_processed'
                                                                    % args.zscore_cutoff)
    ltqs_subset = scdata.originalData.ltqs[:, subset].copy()
    ltqsvars_subset = scdata.originalData.ltqsVars[:, subset].copy()
    mp_print("Storing subset {} with {} cells in folder: {}".format(subset_ind, len(subset),
                                                                    metadata_subset.processedDatafolder))
    storeData(metadata_subset, ltqs_subset, ltqsvars_subset)

if mpi_rank != 0:
    mp_print("The script 'bonsai_iterative_build.py' is not designed to use multiple CPUs in all steps. Currently,"
             "only the preprocessing of the data and creating the subsets of the data benefits from parallelization."
             "Instead, please run the script with the command '--return_commands True'. "
             "It will then return the commands in a text-file "
             "(and printed out on the console) that you can then run yourself using multiple CPUs.",
             ALL_RANKS=True, WARNING=True)
    exit()

"""
------------------------------------------------------------------------------------------
Step 2: Run Bonsai on the first subset.
------------------------------------------------------------------------------------------
"""


