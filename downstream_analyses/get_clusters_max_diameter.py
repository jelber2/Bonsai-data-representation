from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import sys
from downstream_analyses.get_cluster_helpers import Cluster_Tree
from bonsai.bonsai_helpers import mp_print
from itertools import combinations
import csv
import time


# parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# # Add the parent directory to sys.path
# sys.path.append(parent_dir)
# os.chdir(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'tree_layout'))

# def get_cluster_assignments(clusters_list, assign_singlets_together=False):
#     cluster_idx = 0
#     cluster_assigment = []
#     cell_names = []
#     for cluster in clusters_list:
#         # if singleton, assign -1
#         if assign_singlets_together and (len(cluster) == 1):
#             cluster_assigment.append("cl_{}".format(-1))
#             cell_names.append(cluster[0])
#             # fout.write("{}\t{}\n".format(cluster[0], -1))
#         else:
#             for leaf in cluster:
#                 # fout.write("{}\t{}\n".format(leaf, cluster_idx))
#                 cluster_assigment.append("cl_{}".format(cluster_idx))
#                 cell_names.append(leaf)
#             cluster_idx += 1
#
#     # make dict:
#
#     cl_dict = dict(zip(cell_names, cluster_assigment))
#     return cl_dict


def get_cluster_assignments(all_clusterings, node_ids_multiple_cs_ids={}):
    cluster_assignments = {}
    # Create dictionary with dictionaries for each cs-ID, containing for each clustering in which cluster it falls
    for label, clusters in all_clusterings.items():
        for cluster_index, ids in enumerate(clusters):
            for id_ in ids:
                if id_ in node_ids_multiple_cs_ids:
                    for cs_id in node_ids_multiple_cs_ids[id_]:
                        cluster_assignments.setdefault(cs_id, {})[label] = f"cl_{cluster_index}"
                else:
                    cluster_assignments.setdefault(id_, {})[label] = f"cl_{cluster_index}"

    df = pd.DataFrame.from_dict(cluster_assignments, orient="index")
    df = df.sort_index()
    return df


def get_footfall_clustering_from_nwk_str_deprecated(tree_nwk_str, n_clusters, cell_ids=None,
                                                    get_cell_ids_all_splits=False):
    print("\nInit footfall clustering-tree")
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)  # Works
    if get_cell_ids_all_splits:
        clusters, footfall_edges, ids_splits = get_footfall_clustering(cluster_tree, n_clusters, cell_ids=cell_ids,
                                                                       get_cell_ids_all_splits=get_cell_ids_all_splits)
        return clusters, footfall_edges, ids_splits
    else:
        clusters, footfall_edges = get_footfall_clustering(cluster_tree, n_clusters, cell_ids=cell_ids)
    return clusters, footfall_edges


def get_min_pdists_clustering_from_nwk_str(tree_nwk_str, n_clusters, cell_ids=None, get_cell_ids_all_splits=False,
                                             node_id_to_n_cells=None, verbose=True, footfall=False):
    if verbose:
        mp_print("\nInit min-dist clustering-tree")
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)  # Works
    if node_id_to_n_cells is not None:
        cluster_tree.root.add_info_to_nodes(node_id_to_info=node_id_to_n_cells, info_key='n_cells')

    all_clusterings, footfall_edges = get_min_pdists_clustering(cluster_tree, n_clusters, cell_ids=cell_ids,
                                                           verbose=verbose, footfall=footfall)
    return all_clusterings, footfall_edges


def get_min_pdists_clustering_from_nwk_str_deprecated(tree_nwk_str, n_clusters, cell_ids=None,
                                                      get_cell_ids_all_splits=False,
                                                      node_id_to_n_cells=None, verbose=True):
    if verbose:
        print("\nInit min-dist clustering-tree")
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)  # Works
    if node_id_to_n_cells is not None:
        cluster_tree.root.add_info_to_nodes(node_id_to_info=node_id_to_n_cells, info_key='n_cells')
    if get_cell_ids_all_splits:
        clusters, footfall_edges, ids_splits = get_min_pdists_clustering_deprecated(cluster_tree, n_clusters, cell_ids=cell_ids,
                                                                         get_cell_ids_all_splits=get_cell_ids_all_splits)
        return clusters, footfall_edges, ids_splits
    else:
        clusters, footfall_edges = get_min_pdists_clustering_deprecated(cluster_tree, n_clusters, cell_ids=cell_ids,
                                                             verbose=verbose)
    return clusters, footfall_edges


def get_max_diam_clustering_from_nwk_str(tree_nwk_str, max_diam_threshold, cell_ids=None):
    print("\nInit max diameter clustering-tree")
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)  # Works

    clusters = get_max_diam_clustering(cluster_tree, max_diam_threshold, cell_ids=cell_ids)

    return clusters


def get_footfall_clustering(cluster_tree, n_clusters, cell_ids=None, get_cell_ids_all_splits=False):
    if cell_ids is not None:
        cell_id_set = set(cell_ids)
    if get_cell_ids_all_splits:
        cell_ids_splits = {}

    tree_ensmbl = [cluster_tree]
    # Make sure each node knows how many ds leafs it has
    cluster_tree.n_leafs = cluster_tree.root.get_ds_and_parent_info()
    footfall_edges = []

    while len(tree_ensmbl) < n_clusters:
        # Loop over all edges to find max footfall one
        max_footfall_tree_ind = None
        max_footfall_node = None
        max_footfall_score = -1e9
        for ind_tree, tree in enumerate(tree_ensmbl):
            for vert_ind, node in tree.vert_ind_to_node.items():
                if node.parentNode is not None:
                    footfall_score = node.ds_leafs * (tree.n_leafs - node.ds_leafs) * node.tParent
                    if footfall_score > max_footfall_score:
                        max_footfall_tree_ind = ind_tree
                        max_footfall_node = node
                        max_footfall_score = footfall_score

        # Cut the tree into two pieces at the max footfall edge
        ds_node = max_footfall_node
        us_node = max_footfall_node.parentNode
        footfall_edges.append((ds_node.nodeId, us_node.nodeId))

        # Make one new tree:
        max_tree = tree_ensmbl[max_footfall_tree_ind]
        new_tree = Cluster_Tree()

        # Remove ds node from original tree
        us_node.childNodes = [child for child in us_node.childNodes if child.vert_ind != ds_node.vert_ind]

        # Make ds-node the root of the new tree
        new_tree.root = ds_node
        ds_node.parentNode = None

        # tree_1 = max_footfall_tree.copy(minimal_copy=True)
        # tree_2 = max_footfall_tree.copy(minimal_copy=True)
        # Reset the roots of these trees to the connecting nodes
        # tree_1.reset_root(new_root_ind=ds_node_vert_ind)
        # tree_2.reset_root(new_root_ind=us_node_vert_ind)
        # Cut off the redundant parts of the trees
        # tree_1.root.childNodes = [child for child in tree_1.root.childNodes if child.vert_ind != us_node_vert_ind]
        # tree_2.root.childNodes = [child for child in tree_2.root.childNodes if child.vert_ind != ds_node_vert_ind]
        # Update vert_ind_to_node
        max_tree.vert_ind_to_node, max_tree.nNodes = max_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        new_tree.vert_ind_to_node, new_tree.nNodes = new_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        max_tree.n_leafs = max_tree.root.get_ds_and_parent_info()
        new_tree.n_leafs = new_tree.root.get_ds_and_parent_info()

        tree_ensmbl.append(new_tree)
        if get_cell_ids_all_splits:
            cell_ids_splits[(ds_node.nodeId, us_node.nodeId)] = []
            for tree in [max_tree, new_tree]:
                leaf_ids_tree = []
                for vert_ind, node in tree.vert_ind_to_node.items():
                    if cell_ids is None:
                        if node.isLeaf:
                            leaf_ids_tree.append(node.nodeId)
                    else:
                        if node.nodeId in cell_id_set:
                            leaf_ids_tree.append(node.nodeId)
                cell_ids_splits[(ds_node.nodeId, us_node.nodeId)].append(leaf_ids_tree)

    clusters = []
    for ind_tree, tree in enumerate(tree_ensmbl):
        leaf_ids_tree = []
        for vert_ind, node in tree.vert_ind_to_node.items():
            if cell_ids is None:
                if node.isLeaf:
                    leaf_ids_tree.append(node.nodeId)
            else:
                if node.nodeId in cell_id_set:
                    leaf_ids_tree.append(node.nodeId)
        clusters.append(leaf_ids_tree)
    # Should produce a list of lists with the node-IDs of the various clusterings
    print("clustering done")
    if get_cell_ids_all_splits:
        return clusters, footfall_edges, cell_ids_splits
    return clusters, footfall_edges


def get_min_pdists_clustering_deprecated(cluster_tree, n_clusters, cell_ids=None, get_cell_ids_all_splits=False, verbose=True):
    if cell_ids is not None:
        cell_id_set = set(cell_ids)
    if get_cell_ids_all_splits:
        cell_ids_splits = {}

    tree_ensmbl = [cluster_tree]
    # Make sure each node knows how many ds leafs it has
    cluster_tree.n_leafs = cluster_tree.get_min_pdists_info()
    # cluster_tree.n_leafs, cluster_tree.root.ds_dists = cluster_tree.root.get_ds_and_parent_info_plus_dists()
    # cluster_tree.root.us_dists = 0
    # cluster_tree.root.store_us_dists(total_leafs=cluster_tree.n_leafs)
    footfall_edges = []

    while len(tree_ensmbl) < n_clusters:
        # Loop over all edges to find max footfall one
        max_footfall_tree_ind = None
        max_footfall_node = None
        max_footfall_score = -1e9
        for ind_tree, tree in enumerate(tree_ensmbl):
            for vert_ind, node in tree.vert_ind_to_node.items():
                if node.parentNode is not None:
                    footfall_score = node.ds_leafs * (tree.n_leafs - node.ds_leafs) * node.tParent
                    footfall_score += node.ds_dists * (tree.n_leafs - node.ds_leafs)
                    footfall_score += node.us_dists * node.ds_leafs
                    if footfall_score > max_footfall_score:
                        max_footfall_tree_ind = ind_tree
                        max_footfall_node = node
                        max_footfall_score = footfall_score

        # Cut the tree into two pieces at the max footfall edge
        ds_node = max_footfall_node
        us_node = max_footfall_node.parentNode
        footfall_edges.append((ds_node.nodeId, us_node.nodeId))

        # Make one new tree:
        max_tree = tree_ensmbl[max_footfall_tree_ind]
        new_tree = Cluster_Tree()

        # Remove ds node from original tree
        us_node.childNodes = [child for child in us_node.childNodes if child.vert_ind != ds_node.vert_ind]

        # Make ds-node the root of the new tree
        new_tree.root = ds_node
        ds_node.parentNode = None

        # tree_1 = max_footfall_tree.copy(minimal_copy=True)
        # tree_2 = max_footfall_tree.copy(minimal_copy=True)
        # Reset the roots of these trees to the connecting nodes
        # tree_1.reset_root(new_root_ind=ds_node_vert_ind)
        # tree_2.reset_root(new_root_ind=us_node_vert_ind)
        # Cut off the redundant parts of the trees
        # tree_1.root.childNodes = [child for child in tree_1.root.childNodes if child.vert_ind != us_node_vert_ind]
        # tree_2.root.childNodes = [child for child in tree_2.root.childNodes if child.vert_ind != ds_node_vert_ind]
        # Update vert_ind_to_node
        max_tree.vert_ind_to_node, max_tree.nNodes = max_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        new_tree.vert_ind_to_node, new_tree.nNodes = new_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        max_tree.n_leafs = max_tree.get_min_pdists_info()
        new_tree.n_leafs = new_tree.get_min_pdists_info()

        tree_ensmbl.append(new_tree)
        if get_cell_ids_all_splits:
            cell_ids_splits[(ds_node.nodeId, us_node.nodeId)] = []
            for tree in [max_tree, new_tree]:
                leaf_ids_tree = []
                for vert_ind, node in tree.vert_ind_to_node.items():
                    if cell_ids is None:
                        if node.isLeaf:
                            leaf_ids_tree.append(node.nodeId)
                    else:
                        if node.nodeId in cell_id_set:
                            leaf_ids_tree.append(node.nodeId)
                cell_ids_splits[(ds_node.nodeId, us_node.nodeId)].append(leaf_ids_tree)
    # Should produce a list of lists with the node-IDs of the various clusterings
    clusters = []
    for ind_tree, tree in enumerate(tree_ensmbl):
        leaf_ids_tree = []
        for vert_ind, node in tree.vert_ind_to_node.items():
            if cell_ids is None:
                if node.isLeaf:
                    leaf_ids_tree.append(node.nodeId)
            else:
                if node.nodeId in cell_id_set:
                    leaf_ids_tree.append(node.nodeId)
        clusters.append(leaf_ids_tree)
    if verbose:
        print("clustering done")

    if get_cell_ids_all_splits:
        return clusters, footfall_edges, cell_ids_splits
    return clusters, footfall_edges


def get_min_pdists_clustering(cluster_tree, n_clusters, cell_ids=None, get_cell_ids_all_splits=False, verbose=True,
                                  footfall=False):
    # if get_cell_ids_all_splits:
    #     cell_ids_splits = {}
    if cell_ids is not None:
        cell_ids_set = set(cell_ids)
    if cluster_tree.vert_ind_to_node is None:
        cluster_tree.vert_ind_to_node, cluster_tree.nNodes = cluster_tree.root.renumber_verts(vertIndToNode={},
                                                                                              vert_count=0)

    all_clusterings = {}
    tree_ensmbl = [cluster_tree]
    # Make sure each node knows how many ds leafs it has
    cluster_tree.n_leafs = cluster_tree.get_min_pdists_info()
    max_scores = [None]
    max_nodes = [None]
    clusters = [None]
    footfall_edges = []

    # Can't cut more branches between cells than there are cell-associated nodes
    n_clusters = min(n_clusters, cluster_tree.n_cell_nodes)

    print_i = 2
    while len(tree_ensmbl) < n_clusters:
        n_trees = len(tree_ensmbl)
        if verbose and (n_trees == print_i):
            print("Clustering has created {} subtrees, {} branches still to cut.".format(n_trees, n_clusters - n_trees))
            print_i *= 2
        # Loop over all edges to find max footfall one
        for ind_tree, tree in enumerate(tree_ensmbl):
            if max_scores[ind_tree] is not None:
                continue
            if tree.n_cell_nodes == 1:
                max_scores[ind_tree] = 0.0
                continue
            max_footfall_node = None
            max_footfall_score = -1e9
            for vert_ind, node in tree.vert_ind_to_node.items():
                if node.parentNode is not None:
                    footfall_score = node.ds_leafs * (tree.n_leafs - node.ds_leafs) * node.tParent
                    if not footfall:
                        footfall_score += node.ds_dists * (tree.n_leafs - node.ds_leafs)
                        footfall_score += node.us_dists * node.ds_leafs
                    if footfall_score > max_footfall_score:
                        max_footfall_node = node
                        max_footfall_score = footfall_score
            max_scores[ind_tree] = max_footfall_score
            max_nodes[ind_tree] = max_footfall_node

        # Determine which tree has the maximum score
        max_tree_ind = np.argmax(max_scores)
        if max_scores[max_tree_ind] < 1e-9:
            # This should never happen
            print("Cannot find more than {} clusters. Cannot find edge that reduces the pairwise distances.")
            break

        max_node = max_nodes[max_tree_ind]

        # Cut the tree into two pieces at the max footfall edge
        ds_node = max_node
        us_node = max_node.parentNode
        footfall_edges.append((ds_node.nodeId, us_node.nodeId))

        # Make one new tree:
        max_tree = tree_ensmbl[max_tree_ind]
        new_tree = Cluster_Tree()

        # Remove ds node from original tree
        us_node.childNodes = [child for child in us_node.childNodes if child.vert_ind != ds_node.vert_ind]

        # Make ds-node the root of the new tree
        new_tree.root = ds_node
        ds_node.parentNode = None

        # Update vert_ind_to_node
        # TODO: Check at some point whether this can be done efficient by re-using information
        max_tree.vert_ind_to_node, max_tree.nNodes = max_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        new_tree.vert_ind_to_node, new_tree.nNodes = new_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        max_tree.n_leafs = max_tree.get_min_pdists_info()
        new_tree.n_leafs = new_tree.get_min_pdists_info()

        # Add tree to ensemble, and make space for the new tree in the lists
        tree_ensmbl.append(new_tree)
        max_scores[max_tree_ind] = None  # This will make sure the score is recalculated
        max_nodes[max_tree_ind] = None
        clusters[max_tree_ind] = None
        max_scores.append(None)  # This will give a place for the new tree to store
        max_nodes.append(None)
        clusters.append(None)

        # Should produce a list of lists with the node-IDs of the various clusters
        for ind_tree, tree in enumerate(tree_ensmbl):
            if clusters[ind_tree] is not None:
                continue
            leaf_ids_tree = []
            for vert_ind, node in tree.vert_ind_to_node.items():
                if cell_ids is None:
                    if node.isLeaf:
                        leaf_ids_tree.append(node.nodeId)
                else:
                    if node.nodeId in cell_ids_set:
                        leaf_ids_tree.append(node.nodeId)
            clusters[ind_tree] = leaf_ids_tree
        # Store current clustering in dictionary of clusterings
        clustering_name = 'annot_cluster_n{}'.format(len(tree_ensmbl))
        all_clusterings[clustering_name] = clusters.copy()

    if verbose:
        print("clustering done")
    return all_clusterings, footfall_edges


def get_max_diam_clustering(cluster_tree, max_diam_threshold, cell_ids=None):
    # print("\nInit tree")
    # cluster_tree = Cluster_Tree()
    # cluster_tree.from_newick_file(nwk_file=tree_nwk_file)  # Works

    # get post traversal order
    vert_ind_in_postOrder_v3 = cluster_tree.root.getPostOrder_only_internalNodes()

    print("Do clustering with maxdiam: {}".format(max_diam_threshold))
    clusters = []
    # traverse tree in post_order_traversal
    for vert_ind in vert_ind_in_postOrder_v3:
        # print("traversing node: {}".format(vert_ind))
        node = cluster_tree.vert_ind_to_node[vert_ind]

        # if for some reason the node has already been visited...
        if node.is_deleted:
            # print("node {} is deleted".format(node.vert_ind))
            continue

        # Check constraint
        # For that calculate first all pairwise max distances to leaf through parent (node)
        # print("number of children: {}".format(len(node.childNodes)))
        # print("find all combinations of children")

        comb = combinations(range(len(node.childNodes)), 2)
        max_pairwise_distances = {}
        # print("calc max pairwise distances of children")
        for i1, i2 in list(comb):  # store in dict (i1,i2) : dist
            u1 = node.childNodes[i1]
            u2 = node.childNodes[i2]
            if not u1.is_deleted and not u2.is_deleted:
                # TODO : store only if  node is not deleted: done
                dist = u1.len_to_most_distant_leaf + u1.tParent + u2.len_to_most_distant_leaf + u2.tParent
                max_pairwise_distances[(i1, i2)] = dist
        # print("sort max pairwise distances of children")
        # sort distances:
        pair_wise_comb = list(max_pairwise_distances.keys())
        distances = np.array(list(max_pairwise_distances.values()))
        distances_sortidx = np.argsort(-distances)  # largest will be at first position

        # check if all max pairwise distances fullfill the constraint
        # for that I can check if the largest max pairwise distance fullfills the constraint
        # print("len(pair_wise_comb): {}".format(len(pair_wise_comb)))
        # print("start while loop")
        while len(pair_wise_comb) > 0:
            # print("len(pair_wise_comb): {}".format(len(pair_wise_comb)))
            # print("check if contraint is met for the largest distances between two leafs")
            if distances[distances_sortidx][0] < max_diam_threshold:
                # print("contraint is met, we can continue")
                longest_distances_to_leafs_from_node = [x.len_to_most_distant_leaf + x.tParent for x in node.childNodes
                                                        if not x.is_deleted]
                node.len_to_most_distant_leaf = np.max(longest_distances_to_leafs_from_node)
                break
            else:
                # print("contraint is NOT met, we have to cut the longest branch")
                # cut longest branch

                i1, i2 = np.array(pair_wise_comb)[distances_sortidx][0]
                u1 = node.childNodes[i1]
                u2 = node.childNodes[i2]
                # cut longest branch and get cluster
                if u1.len_to_most_distant_leaf + u1.tParent >= u2.len_to_most_distant_leaf + u2.tParent:
                    # print("longest branch corresponds to child: {}".format(i1))
                    cluster = cluster_tree.cut(subtree_root=u1, cell_ids=cell_ids)
                    # todo set u1 to deleted
                    deleted_node_idx = i1
                    clusters.append(cluster)
                    node.len_to_most_distant_leaf = u2.len_to_most_distant_leaf + u2.tParent
                else:
                    # print("longest branch corresponds to child: {}".format(i2))
                    cluster = cluster_tree.cut(subtree_root=u2, cell_ids=cell_ids)
                    deleted_node_idx = i2
                    clusters.append(cluster)
                    node.len_to_most_distant_leaf = u1.len_to_most_distant_leaf + u1.tParent

                # print("Now remove all that corresponds to that longest branch and redo while loop")
                # then delete the node and subtree of the longest branch, and check again for the distances

                valid_pairs = [True if x[0] != deleted_node_idx and x[1] != deleted_node_idx else False for x in
                               pair_wise_comb]

                pair_wise_comb = [b for a, b in zip(valid_pairs, pair_wise_comb) if a]
                distances = distances[valid_pairs]
                distances_sortidx = np.argsort(
                    -distances)  # i think i have to sort new TODO check if this is right, make more efficient

    # at the end get the rest of the cluster
    cluster = cluster_tree.cut(subtree_root=node, cell_ids=cell_ids)
    clusters.append(cluster)
    print("clustering done")
    return clusters


if __name__ == "__main__":

    parser = ArgumentParser(
        description='Starts from a reconstructed tree output by Bonsai and creates a data-object necessary for further '
                    'visualization and usage in the Bonsai-Shiny app.')

    parser.add_argument('--dataset', type=str, default='test_sarah',
                        help='Name of dataset. This will determine name of results-folder where information is stored.')
    # Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
    # parser.add_argument('--results_folder', type=str, default="/Users/sarahmorillo/bz_mnt/software/waddington-code-github/python_waddington_code/downstream_analyses/test_res",
    #                     help='path to folder where results will be stored.')
    parser.add_argument('--tree_folder', type=str,
                        # default='/Users/sarahmorillo/bz_mnt/software/waddington-code-github/python_waddington_code/downstream_analyses/test_data/tamara_ecoli_isolates/mergers_zscore1.0_ellipsoidsize1.0_smallerrorbars_redoStarry_optTimes_nnnReorder_reorderedEdges',
                        default='/Users/sarahmorillo/bz_mnt/software/waddington-code-github/downstream_analyses/test/test_data/',
                        help='Path to folder that determines tree topology. Should contain edgeInfo and vertInfo and tree.nwk')
    parser.add_argument('--output_file', type=str,
                        default="/Users/sarahmorillo/bz_mnt/software/waddington-code-github/downstream_analyses/test/test_res/test_binary_tree_8_leafs",
                        help="output file name")
    parser.add_argument('--t', type=float, default=5, dest='max_diam_threshold', help="max diam threshold")
    # Arguments that determine running configurations of MLTree. How much is printed, which steps are run?
    # parser.add_argument('--verbose', type=str2bool, default=True,
    #                     help='--verbose False only shows essential print messages (default: True)')

    args = parser.parse_args()
    print(args)

    """Test max footfall and min pdists clustering on some test case"""
    nwk_file = '/Users/Daan/Documents/postdoc/bonsai-development/results/simulated_datasets/simulated_binary_6_gens_samplingnoise_seed_1231/final_bonsai_zscore1.0/tree.nwk'
    with open(nwk_file, "r") as f:
        nwk_str = f.readline()
    clusters_list, cut_edges = get_footfall_clustering_from_nwk_str(tree_nwk_str=nwk_str,
                                                                    n_clusters=8)
    clusters_list_min_pd, cut_edges_min_pd = get_min_pdists_clustering_from_nwk_str(tree_nwk_str=nwk_str,
                                                                                    n_clusters=8)

    """In the following lines the tree is read in from the output generated by Bonsai"""

    clusters_md = get_max_diam_clustering_from_nwk_file(
        tree_nwk_file=os.path.join(args.tree_folder, 'test_binary_tree_8_leafs.nwk'),
        max_diam_threshold=args.max_diam_threshold)

    cl_dict = get_cluster_assignments(clusters_list=clusters_md)

    # print output:
    print("number of clusters found: {}".format(len(clusters_md)))
    # write to file:
    with open(args.output_file + "-max_diam_{}.cluster".format(args.max_diam_threshold), "w") as fout:
        cluster_idx = 0
        for cluster in clusters_md:
            # if singleton, assign -1
            if len(cluster) == 1:
                fout.write("{}\t{}\n".format(cluster[0], -1))
            else:
                for leaf in cluster:
                    fout.write("{}\t{}\n".format(leaf, cluster_idx))

                cluster_idx += 1

    print("done")

    # try out to get longest path from root to a leaf
    cluster_tree_md = Cluster_Tree()
    cluster_tree_md.from_newick_file(
        nwk_file=os.path.join(args.tree_folder, 'test_binary_tree_8_leafs-longest_path_to_leaf_11.5.nwk'))  # Works
    longest_path_from_root_to_leaf, _ = cluster_tree_md.root.find_longest_path_between_two_leafs()
