import numpy as np
import os
import sys
import h5py
from pathlib import Path
import pandas as pd
from scipy.special import logsumexp
from argparse import ArgumentParser
import subprocess

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

from bonsai.bonsai_helpers import write_ids

parser = ArgumentParser(
    description='Simulates a binary tree in a lower-dimensional space.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--results_folder', type=str, default='data/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder for storing simulated data.")
parser.add_argument('--num_rows', type=int, default=512,
                    help="Number of generations in binary tree.")
parser.add_argument('--num_cols', type=int, default=1024,
                    help="Number of generations in binary tree.")

args = parser.parse_args()
print(args)

seed = 2462
np.random.seed(seed)

# Set number of generations and diffusion times
datadir = "random_numbers_{}genes_{}cells_seed{}".format(args.num_rows, args.num_cols, seed)
gene_ids = ['Gene_' + str(ind) for ind in range(args.num_rows)]
cell_ids = ['Cell_' + str(ind) for ind in range(args.num_cols)]
data_path = os.path.abspath(os.path.join(args.results_folder, datadir))
Path(data_path).mkdir(parents=True, exist_ok=True)

print("Writing cell IDs to file:")
write_ids(os.path.join(data_path, 'cellID.txt'), cell_ids)
write_ids(os.path.join(data_path, 'geneID.txt'), gene_ids)

print("Sampling UMI counts:")
umi_counts = np.random.randint(low=0, high=1001, size=(args.num_rows, args.num_cols))

print("Writing UMI counts to file:")
umi_df = pd.DataFrame(umi_counts, columns=cell_ids, index=gene_ids)
umi_df.to_csv(os.path.join(data_path, 'Gene_table.txt'), sep='\t', index_label="GeneID")

umi_df.to_csv(os.path.join(data_path, 'features.txt'), sep='\t', header=False, index=False)

print("Writing celltypes to file:")
annotation_dict = {}
# Add total UMI-counts per cell as annotation
total_counts_c = list(np.sum(umi_counts, axis=0))
annotation_dict['total_count'] = total_counts_c

annotation_df = pd.DataFrame(annotation_dict, index=cell_ids)
Path(os.path.join(data_path, 'annotation')).mkdir(parents=True, exist_ok=True)
annotation_df.to_csv(os.path.join(data_path, 'annotation', 'minimal_annotation.csv'))
