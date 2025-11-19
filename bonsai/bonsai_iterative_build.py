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
    description='Starts from cell-data and a tree on which some of the cells are already placed. Cells that are not on'
                'the tree but are in the data will be placed one by one on the tree.')

parser.add_argument('--config_filepath', type=str, default=None,
                    help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
                         'needed to run Bonsai.')

# TODO: To be moved to config-file
parser.add_argument('--select_target', type=str, default='cluster_centers',
                    help="This will determine what strategy we follow for the "
                         "spr-moves. Current options are 'root', 'cluster_centers', 'exhaustive'")

parser.add_argument('--search_tol', type=float, default=2,
                    help="Gives the loglikelihood-margin by which a new SPR-search has to be worse than a previous one"
                         "before we discard this search direction")

parser.add_argument('--iterative_cell_lists', type=str, default=None,
                    help="[NOT YET IMPLEMENTED]"
                         "One can give a filename here for a file where we have all cell-IDs (one per line) in the "
                         "first column. Second column contains an integer indicating at which iteration the cells "
                         "should be added. Useful for modeling time-course data.")

parser.add_argument('--n_initial_cells', type=int, default=10000,
                    help="If --iterative_cell_lists is None, this number will determine the size of the subset of "
                         "cells on which we will make the initial guide-tree.")

parser.add_argument('--growth_before_cleanup', type=float, default=.5,
                    help="When adding cells, this factor (larger than 0) determines after what growth factor (so .1 "
                         "means 10% growth) we re-optimize the tree before adding another set of cells.")

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

# TODO: Make sure that Run_configs initialization just copies all arguments that were originally in args
pickup_intermediate = args.pickup_intermediate
growth_before_cleanup = args.growth_before_cleanup
select_target = args.select_target
iterative_cell_lists = args.iterative_cell_lists
n_initial_cells = args.n_initial_cells
return_commands = args.return_commands
config_filepath = args.config_filepath
seed = args.seed
search_tol = args.search_tol

args = Run_Configs(args.config_filepath)

args.search_tol = search_tol
args.select_target = select_target
args.iterative_cell_lists = iterative_cell_lists
args.growth_before_cleanup = growth_before_cleanup
args.n_initial_cells = n_initial_cells
args.return_commands = return_commands
args.config_filepath = config_filepath
if pickup_intermediate:
    args.pickup_intermediate = True

import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_dataprocessing import initializeSCData, getMetadata, loadReconstructedTreeAndData, SCData, \
    nnnReorder, nnnReorderRandom, Metadata, storeData
from bonsai.bonsai_helpers import mp_print, startMPI, getOutputFolder, get_latest_intermediate, \
    clean_up_redundant_data_files, str2bool
from bonsai.bonsai_treeHelpers import Tree
import bonsai.bonsai_globals as bs_glob

# Some of the code can be run in parallel, parallelization is done via mpi4py
mpi_rank, mpi_size = startMPI(verbose=True)

start_all = time.time()

if mpi_rank != 0:
    mp_print("The script 'bonsai_iterative_build.py' is not designed to use multiple CPUs in all steps."
             "Instead, please run this script with the command '--return_commands True'. "
             "It will then return the commands in a text-file "
             "(and printed out on the console) that you can then run yourself using multiple CPUs.",
             ALL_RANKS=True, WARNING=True)
    exit()

"""
------------------------------------------------------------------------------------------
Step 1: Get a subset of the cells and store their information in a subfolder.
------------------------------------------------------------------------------------------
"""

scdata = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
if args.return_commands:
    commands_file = scdata.result_path('iterative_build_commands.txt')

scdata.metadata.pathToOrigData = args.data_folder
# Get cell-IDs of whole dataset
scdata.metadata.cellIds = []
if os.path.exists(scdata.data_path('cellID.txt')):
    with open(scdata.data_path('cellID.txt'), 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            scdata.metadata.cellIds.append(row[0])
    scdata.metadata.nCells = len(scdata.metadata.cellIds)
    n_unq_ids = len(np.unique(scdata.metadata.cellIds))
    if scdata.metadata.nCells is not None:
        if n_unq_ids != scdata.metadata.nCells:
            mp_print("Cell-IDs are not unique.\n "
                     "Please resolve this and start this script again.", ERROR=True)
            exit()

scdata.metadata.processedDatafolder = scdata.result_path('zscorefiltered_%.3f_and_processed' % args.zscore_cutoff)

# Select a subset of the cell-IDs to build a first guide-tree on
np.random.seed(seed)

n_initial_cells = min(args.n_initial_cells, scdata.metadata.nCells)
subset = np.random.choice(np.arange(scdata.metadata.nCells), size=n_initial_cells, replace=False)
cell_ids_subset = [scdata.metadata.cellIds[ind] for ind in subset]

# Also create a scdata-object for the subset
metadata_subset = Metadata(curr_metadata=scdata.metadata)
metadata_subset.cellIds = cell_ids_subset
metadata_subset.nCells = len(cell_ids_subset)
bs_glob.nCells = metadata_subset.nCells
metadata_subset.dataset = os.path.join(scdata.metadata.dataset, 'subset_0')
metadata_subset.results_folder = os.path.join(scdata.metadata.results_folder, 'subset_0')

scdata_subset = SCData(onlyObject=True, dataset=metadata_subset.dataset,
                       results_folder=metadata_subset.results_folder)

scdata_subset.metadata = metadata_subset
scdata_subset.metadata.processedDatafolder = scdata_subset.result_path('zscorefiltered_%.3f_and_processed' % args.zscore_cutoff)

# Store this subset in a file such that the other script can read it in.
mp_print("Writing cell IDs to file:")
with open(scdata.result_path('starting_cell_ids.txt'), 'w') as f:
    for ID in cell_ids_subset:
        f.write("%s\n" % ID)

# Run the script that will read in and preprocess the data, and create the star-tree for the first guide-tree
preprocess_cmd = ['bonsai/bonsai_iterative_first_split.py',
                  '--config_filepath', args.config_filepath,
                  '--starting_cell_ids', scdata.result_path('starting_cell_ids.txt')]

if not args.return_commands:
    output1 = subprocess.run([sys.executable] + preprocess_cmd, stdout=subprocess.PIPE, text=True)
    mp_print(output1.stdout)
    mp_print(output1.stderr)
else:
    cmd = ' '.join(preprocess_cmd)
    mp_print("Command for creating initial Bonsai guide-tree (stored in {}):\n "
          "{}".format(commands_file, cmd))
    with open(commands_file, "w") as file:
        file.write(cmd + '\n')


"""
------------------------------------------------------------------------------------------
Step 2: Run Bonsai on the first subset.
------------------------------------------------------------------------------------------
"""


def get_command_config_file(args, dataset, result_path, data_folder=None, tmp_folder=None):
    config_filepath = os.path.join(result_path, 'iterative_configs_yaml.yaml')
    if data_folder is None:
        data_folder = result_path
    command1 = [sys.executable,
                'bonsai/create_config_file.py',
                '--new_yaml_path',
                config_filepath,
                '--dataset',
                dataset,
                '--data_folder',
                data_folder,
                '--results_folder',
                result_path,
                '--input_is_sanity_output',
                str(args.input_is_sanity_output),
                '--zscore_cutoff',
                str(args.zscore_cutoff),
                '--UB_ellipsoid_size',
                str(args.UB_ellipsoid_size),
                '--nnn_n_randommoves',
                str(args.nnn_n_randommoves),
                '--nnn_n_randomtrees',
                str(args.nnn_n_randomtrees),
                '--pickup_intermediate',
                'True',
                '--use_knn',
                str(args.use_knn),
                '--rescale_by_var',
                str(args.rescale_by_var)]
    if tmp_folder is not None:
        command1 += ['--tmp_folder', tmp_folder]
    return command1, config_filepath


# Create new config-file for the subset
create_config_command_1, config_filepath = get_command_config_file(args, scdata_subset.metadata.dataset,
                                                                   scdata_subset.result_path())
output1 = subprocess.run(create_config_command_1, stdout=subprocess.PIPE, text=True)
mp_print(output1.stdout)
mp_print(output1.stderr)

# Run Bonsai on the new config-file
bonsai_subset1_cmd = ['bonsai/bonsai_main.py',
                      '--config_filepath', config_filepath,
                      '--step', 'all',
                      '--spr_strategy', 'large_tree',
                      '--pickup_intermediate', 'True']

if not args.return_commands:
    output1 = subprocess.run([sys.executable] + bonsai_subset1_cmd, stdout=subprocess.PIPE, text=True)
    mp_print(output1.stdout)
    mp_print(output1.stderr)
else:
    cmd = ' '.join(bonsai_subset1_cmd)
    mp_print("Command for creating initial Bonsai guide-tree (stored in {}):\n "
          "{}".format(commands_file, cmd))
    with open(commands_file, "a") as file:
        file.write(cmd + '\n')

results_dir_subset0 = scdata_subset.result_path(getOutputFolder(zscore_cutoff=args.zscore_cutoff, final=True))

"""
------------------------------------------------------------------------------------------
Step 3: Add remaining cells to the initial guide-tree.
------------------------------------------------------------------------------------------
"""
# TODO: Eventually allow looping over remaining subsets

final_dataset_name = os.path.join(scdata.metadata.dataset, 'subset_final')
final_result_folder = os.path.join(scdata.result_path(), 'subset_final')
create_config_command_1, config_filepath = get_command_config_file(args, final_dataset_name, final_result_folder,
                                                                   data_folder=args.data_folder,
                                                                   tmp_folder=scdata_subset.result_path())
output2 = subprocess.run(create_config_command_1, stdout=subprocess.PIPE, text=True)
mp_print(output1.stdout)
mp_print(output1.stderr)

add_cells_seed = np.random.randint(1e6)
add_cells_cmd = ['bonsai/bonsai_add_cells.py',
                 '--config_filepath', config_filepath,
                 '--guide_tree_folder', results_dir_subset0,
                 '--preprocessed_data_folder', scdata.metadata.processedDatafolder,
                 '--growth_before_cleanup', str(args.growth_before_cleanup),
                 '--select_target', args.select_target,
                 '--search_tol', str(args.search_tol),
                 '--seed', str(add_cells_seed)]
# TODO: Add arguments '--nodes_to_add_to', '--cels_to_be_added' later

if not args.return_commands:
    output1 = subprocess.run([sys.executable] + add_cells_cmd, stdout=subprocess.PIPE, text=True)
    mp_print(output1.stdout)
    mp_print(output1.stderr)
else:
    cmd = ' '.join(add_cells_cmd)
    mp_print("Command for adding remaining cells to guide tree (stored in {}):\n{}".format(commands_file, cmd))
    with open(commands_file, "a") as file:
        file.write(cmd + '\n')

"""
------------------------------------------------------------------------------------------
Step 4: Run Bonsai one more time starting from the created tree to see what spr-moves and NNI moves are still possible
------------------------------------------------------------------------------------------
"""

create_config_command_1, config_filepath = get_command_config_file(args, scdata.metadata.dataset, scdata.result_path(),
                                                                   tmp_folder=final_result_folder)
output2 = subprocess.run(create_config_command_1, stdout=subprocess.PIPE, text=True)
mp_print(output1.stdout)
mp_print(output1.stderr)

# Run Bonsai on the new config-file
bonsai_subset1_cmd = ['bonsai/bonsai_main.py',
                      '--config_filepath', config_filepath,
                      '--step', 'all',
                      '--spr_strategy', 'large_tree',
                      '--pickup_intermediate', 'True']

if not args.return_commands:
    output1 = subprocess.run([sys.executable] + bonsai_subset1_cmd, stdout=subprocess.PIPE, text=True)
    mp_print(output1.stdout)
    mp_print(output1.stderr)
else:
    cmd = ' '.join(bonsai_subset1_cmd)
    mp_print("Command for creating initial Bonsai guide-tree (stored in {}):\n "
          "{}".format(commands_file, cmd))
    with open(commands_file, "a") as file:
        file.write(cmd + '\n')

print_text = 'only printing commands' if args.return_commands else "performing all calculations"
mp_print("Running the iterative-bonsai script while {} took {} seconds.".format(print_text, time.time() - start_all))
