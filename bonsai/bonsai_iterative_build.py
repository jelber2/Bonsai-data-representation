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
                         "spr-moves. Current options are 'root', 'cluster_centers', 'exhaustive', 'leafs', "
                         "'leafs_exhaustive'.")

parser.add_argument('--cells_to_be_added', type=str, default=None,
                    help="One can give a filename to a txt-file containing cell-IDs here (one per line). In that case,"
                         "only these cell-IDs will be added.")

args = parser.parse_args()
select_target = args.select_target
cells_to_be_added = args.cells_to_be_added

args = Run_Configs(args.config_filepath)
args.select_target = select_target

import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_dataprocessing import initializeSCData, getMetadata, loadReconstructedTreeAndData, SCData, \
    nnnReorder, nnnReorderRandom
from bonsai.bonsai_helpers import mp_print, startMPI, getOutputFolder, get_latest_intermediate, \
    clean_up_redundant_data_files, str2bool
import bonsai.bonsai_globals as bs_glob

# Some of the code can be run in parallel, parallelization is done via mpi4py
mpiRank, mpiSize = startMPI(args.verbose)
origEllipsoidSize = None

start_all = time.time()