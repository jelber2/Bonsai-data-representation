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

args = parser.parse_args()
select_target = args.select_target
iterative_cell_lists = args.iterative_cell_lists
n_initial_cells = args.n_initial_cells

args = Run_Configs(args.config_filepath)
args.select_target = select_target
args.iterative_cell_lists = iterative_cell_lists
args.n_initial_cells = n_initial_cells

import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_dataprocessing import initializeSCData, getMetadata, loadReconstructedTreeAndData, SCData, \
    nnnReorder, nnnReorderRandom
from bonsai.bonsai_helpers import mp_print, startMPI, getOutputFolder, get_latest_intermediate, \
    clean_up_redundant_data_files, str2bool
import bonsai.bonsai_globals as bs_glob

# Some of the code can be run in parallel, parallelization is done via mpi4py
mpi_rank, mpi_size = startMPI(args.verbose)
if mpi_rank != 0:
    exit()

start_all = time.time()
