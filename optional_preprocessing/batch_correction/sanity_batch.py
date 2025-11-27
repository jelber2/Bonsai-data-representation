from argparse import ArgumentParser
import os
import sys
import pandas as pd
import csv
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from pathlib import Path
import logging
import psutil
import subprocess

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from bonsai.bonsai_helpers import mp_print, str2bool, Run_Configs

parser = ArgumentParser(description='Splits a dataset into batches, and starts Sanity runs separately per batch.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--count_file', type=str, default=None,
                    help='Absolute path to file where counts can be found, should be either a .tsv- or a .mtx-file.')
parser.add_argument('--batch_annotation_file', type=str, default=None,
                    help='Absolute path to tsv-file with two columns. Column 1 contains all cell-IDs, column2 contains'
                         'an identifier of the batch that cell belongs to.')
parser.add_argument('--gene_names_file', type=str, default=None,
                    help='(OPTIONAL): Absolute path to file where gene-IDs are stored (one per line).')
parser.add_argument('--cell_names_file', type=str, default=None,
                    help='(OPTIONAL): Absolute path to file where cell-IDs are stored (one per line).')
parser.add_argument('--results_folder', type=str, default=None,
                    help='Absolute path to where the final results should be stored.')
parser.add_argument('--sanity_binary_path', type=str, default=None,
                    help='Absolute path to compiled Sanity binary')
parser.add_argument('--conda_env', type=str, default=None,
                    help='(OPTIONAL) Name of Sanity conda-environment')
parser.add_argument('--verbose', type=str2bool, default=False,
                    help='--verbose False only shows essential print messages (default: True)')

args = parser.parse_args()
mp_print(args)

SKIP_READING = False
SKIP_SANITY = False

FORMAT = '%(asctime)s %(name)s %(funcName)s %(message)s'
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.getLogger().setLevel(log_level)


"""---------------------------Read in the raw counts.---------------------------"""
logging.info("Reading in data")
input_folder = os.path.dirname(os.path.abspath(args.count_file))

# Read in the batch-information
batch_annotation = pd.read_csv(args.batch_annotation_file, sep='\t', index_col=0, header=None)
# batches = list(batch_annotation.loc[:, 0])
batches = list(batch_annotation.iloc[:, 0])
batch_ids, cell_ind_to_batch_ind, counts = np.unique(batches, return_inverse=True, return_counts=True)
n_batches = len(batch_ids)

if not SKIP_READING:
    if args.count_file.split('.')[1] == 'mtx':
        logging.debug("Reading in raw data from .mtx-file")
        M = mmread(args.count_file)
        logging.debug("Done reading in raw data from .mtx-file")
        umi_counts = M.toarray()
        # Read in promoter names
        gene_ids = []
        with open(os.path.join(args.gene_names_file), 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                gene_ids.append(row[0])

        # Read in cell barcodes as in mtx-file
        cell_ids = []
        with open(args.cell_names_file, 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                cell_ids.append(row[0])
    else:
        logging.debug("Reading in raw data from .txt-file")
        tmp = pd.read_csv(args.count_file, sep='\t', index_col=0)
        logging.debug("Done reading in raw data from .txt-file")
        cell_ids = list(tmp.columns)
        gene_ids = list(tmp.index)
        umi_counts = tmp.values

        # sparse_umis = csr_matrix(umi_counts)
        # mmwrite(os.path.join(input_folder, 'prom_expr_matrix.mtx'), sparse_umis)
        #
        # print("Writing cell IDs to file:")
        # with open(os.path.join(input_folder, 'cellID.txt'), 'w') as f:
        #     for ID in cell_ids:
        #         f.write("%s\n" % ID)
        #
        # print("Writing gene IDs to file:")
        # with open(os.path.join(input_folder, 'geneID.txt'), 'w') as f:
        #     for ID in gene_ids:
        #         f.write("%s\n" % ID)

    n_genes, n_cells = umi_counts.shape

    """---------------------------Split up the counts according to batch-annotation.---------------------------"""
    logging.info("Splitting data in batches")
    cell_id_to_batch_id = {cell_ids[ind]: batch_ids[cell_ind_to_batch_ind[ind]] for ind in range(n_cells)}

    batch_cell_inds = {}
    batch_cell_ids = {}
    batch_n_cells = {}
    for ind_batch, batch_id in enumerate(batch_ids):
        n_cells_batch = counts[ind_batch]
        batch_cell_inds[batch_id] = []
        batch_cell_ids[batch_id] = []
        batch_n_cells[batch_id] = 0

    logging.debug("Splitting the counts into batches.")
    # TODO: Speed this up
    for cell_ind, cell_id in enumerate(cell_ids):
        batch_id = cell_id_to_batch_id[cell_id]
        # nth_cell = batch_n_cells[batch_id]
        batch_cell_inds[batch_id].append(cell_ind)
        batch_cell_ids[batch_id].append(cell_id)
        # batch_n_cells[batch_id] += 1

    batch_counts = {}
    for batch_id in batch_ids:
        batch_counts[batch_id] = umi_counts[:, np.array(batch_cell_inds[batch_id])]

    logging.debug("Done splitting the counts into batches.")

    # Store the data per batch in subfolders
    logging.debug("Storing the batch-counts.")
    for batch_id in batch_ids:
        Path(os.path.join(input_folder, 'batch_counts', batch_id)).mkdir(parents=True, exist_ok=True)
        sparse_umis = csr_matrix(batch_counts[batch_id])
        mmwrite(os.path.join(input_folder, 'batch_counts', batch_id, 'prom_expr_matrix.mtx'), sparse_umis)

        with open(os.path.join(input_folder, 'batch_counts', batch_id, 'accepted_barcodes.tsv'), 'w') as f:
            for ID in batch_cell_ids[batch_id]:
                f.write("%s\n" % ID)

        with open(os.path.join(input_folder, 'batch_counts', batch_id, 'prom_expr_promoters.tsv'), 'w') as f:
            for ID in gene_ids:
                f.write("%s\n" % ID)
    logging.debug("Done storing the batch-counts.")

"""---------------------------Run Sanity on different batches separately.---------------------------"""
if not SKIP_SANITY:
    logging.info("Running Sanity on batches")

    if args.conda_env is not None:
        cmd = ['conda', 'run', '-n', args.conda_env, "--no-capture-output"]
    else:
        cmd = []

    # Use SLURM allocation if available; otherwise get physical cores
    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=False)))

    for batch_id in batch_ids:
        batch_folder = os.path.join(input_folder, 'batch_counts', batch_id)
        matrix_file = os.path.join(batch_folder, 'prom_expr_matrix.mtx')
        genes_file = os.path.join(batch_folder, 'prom_expr_promoters.tsv')
        cells_file = os.path.join(batch_folder, 'accepted_barcodes.tsv')

        sanity_output_folder = os.path.join(args.results_folder, batch_id)
        logfile = os.path.join(sanity_output_folder, 'sanity_log.txt')
        Path(sanity_output_folder).mkdir(parents=True, exist_ok=True)
        cmd += [args.sanity_binary_path,
               "-f", matrix_file,
               '-mtx_genes', genes_file,
               '-mtx_cells', cells_file,
               '-d', sanity_output_folder,
               '-n', str(total_cpus),
               '-e', '1',
               '-max_v', 'only_max_output']

        logging.info("Starting Sanity on batch {}.".format(batch_id))
        logging.debug(" ".join(cmd))
        logging.info("Logging Sanity-output in {}\n".format(logfile))

        with open(logfile, "w") as logf:
            # Start the process
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            )

            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                logging.debug(line.rstrip("\n"))

            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
        # subprocess.run(cmd, check=True)

"""---------------------------For each Sanity result, divide out prior independently.---------------------------"""

"""---------------------------Compile the Sanity results into one feature-matrix.---------------------------"""

"""---------------------------Create bonsai-YAML-config-file for the Bonsai run.---------------------------"""
