import pandas as pd
import numpy as np


def get_batch_annotation(batch_annotation_file):
    # Read in the batch-information
    batch_annotation = pd.read_csv(batch_annotation_file, sep='\t', index_col=0, header=None)
    # batches = list(batch_annotation.loc[:, 0])
    batches = list(batch_annotation.iloc[:, 0])
    cell_ids_annot = list(batch_annotation.index)
    batch_ids, cell_ind_to_batch_ind, counts = np.unique(batches, return_inverse=True, return_counts=True)
    batch_ids = list(batch_ids)
    n_cells = len(cell_ids_annot)
    cell_id_to_batch_id = {cell_ids_annot[ind]: batch_ids[cell_ind_to_batch_ind[ind]] for ind in range(n_cells)}

    return batch_ids, cell_ind_to_batch_ind, cell_id_to_batch_id, counts