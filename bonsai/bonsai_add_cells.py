from argparse import ArgumentParser
import numpy as np
import time
from pathlib import Path
import os, sys

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
parser.add_argument('--spr_strategy', type=str, default='cluster_centers',
                    help="This will determine what strategy we follow for the "
                         "spr-moves. Current options are 'root', 'cluster_centers', 'exhaustive', 'leafs', "
                         "'leafs_exhaustive'.")

parser.add_argument('--cells_to_be_added', type=str, default=None,
                    help="One can give a filename to a txt-file containing cell-IDs here (one per line). In that case,"
                         "only these cell-IDs will be added.")

args = parser.parse_args()
spr_strategy = args.spr_strategy

args = Run_Configs(args.config_filepath)
args.spr_strategy = spr_strategy

import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_dataprocessing import initializeSCData, getMetadata, loadReconstructedTreeAndData, SCData, \
    nnnReorder, nnnReorderRandom
from bonsai.bonsai_helpers import mp_print, startMPI, getOutputFolder, get_latest_intermediate, \
    clean_up_redundant_data_files, str2bool
import bonsai.bonsai_globals as bs_glob

# Some of the code can be run in parallel, parallelization is done via mpi4py
mpiRank, mpiSize = startMPI(args.verbose)
origEllipsoidSize = None

# The SEQUENTIAL-variables determines whether we first optimise diff. times between merged nodes, and then to ancestor
# from root, or all three at the same time. SEQUENTIAL=True is faster and leads to better tree likelihoods in tests
SEQUENTIAL = False

start_all = time.time()

# Read in the tree that was created on a subset of the data (read in without data)
scdata_tmp = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
results_folder = scdata_tmp.result_path()

output_folder = find_latest_tree_folder_new(args, results_folder, not_final=True)
all_genes = False

scdata_guide = loadReconstructedTreeAndData(args, output_folder, all_genes=all_genes, get_cell_info=False,
                                      reprocess_data=False, all_ranks=True, rel_to_results=True, get_data=False,
                                      no_data_needed=True, get_posterior_ltqs=False, otherRanksMinimalInfo=True)

# Read in the data for all cells, and preprocess it like normal
scdata_all_cells = initializeSCData(args, createStarTree=False, getOrigData=False, otherRanksMinimalInfo=True)
cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(scdata_all_cells.metadata.cellIds)}

# Create list of cells that is already on the tree, and get cell-ID to node dictionary to be able to add ltq-info
nodes_list = scdata_guide.tree.root.getNodeList([], returnRoot=True, returnLeafs=True)

# The gene-selection will be based on all cells, so we copy the metadata based on scdata_all_cells
scdata_guide.metadata.geneIds = scdata_all_cells.metadata.geneIds
scdata_guide.metadata.geneVariances = scdata_all_cells.metadata.geneVariances
scdata_guide.metadata.loglikVarCorr = scdata_all_cells.metadata.loglikVarCorr
scdata_guide.metadata.pathToOrigData = scdata_all_cells.metadata.pathToOrigData
scdata_guide.metadata.processedDatafolder = scdata_all_cells.metadata.processedDatafolder
scdata_guide.metadata.results_folder = scdata_all_cells.metadata.results_folder

# Add data to guide-tree, take signal-to-noise genes based on all cells
guide_cell_inds = []
for node in nodes_list:
    if node.nodeId in cell_id_to_ind:
        # In this case, the node corresponds to a cell. We're assuming that each node has max 1 cell
        cell_ind = cell_id_to_ind[node.nodeId]
        guide_cell_inds.append(cell_ind)
        node.ltqs = scdata_all_cells.originalData.ltqs[:, cell_ind]
        node.setLtqsVarsOrW(ltqsVars=scdata_all_cells.originalData.ltqsVars[:, cell_ind])
        node.isCell = True

guide_cell_inds = np.unique(guide_cell_inds)
non_guide_cell_inds = np.setdiff1d(np.arange(scdata_all_cells.metadata.nCells), guide_cell_inds)

# Make sure that all ltqs are calculated at all nodes (automatically done when calculating a loglikelihood)
scdata_guide.metadata.loglik = scdata_guide.tree.calcLogLComplete(mem_friendly=True,
                                                      loglikVarCorr=scdata_guide.metadata.loglikVarCorr)
mp_print("Loglikelihood of guide tree before adding cells: " + str(scdata_guide.metadata.loglik))

# Create random order of cells to be added
np.random.shuffle(non_guide_cell_inds)
ltqs_to_add = scdata_all_cells.originalData.ltqs[:, non_guide_cell_inds]
ltqsvars_to_add = scdata_all_cells.originalData.ltqsVars[:, non_guide_cell_inds]

# TODO: Store these two data-matrices in an .npy-file, such that we can do lazy reading. Then just give the filenames
scdata_guide.add_cells(ltqs_to_add, ltqsvars_to_add)



# Calculate cluster-centers
# Loop over the remaining cells (in a random order), and add them to the tree using cluster centers
# Keep track of how many cells attach to the root (indicating that their celltype was not represented yet?)

# Every N cells we postprocess: re-optimize branch lengths, resolve-polytomies, re-optimize branch lengths,
# and calculate new cluster-centers
# Maybe N should be 100-1000, or it should scale with the number of cells already in the tree (after ~10% increase)

# Do this until all cells are added

# Still do SPR-moves and NNI-moves maybe.

# Store like normal



