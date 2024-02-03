import itertools as it
from collections import defaultdict

import networkx as nx
import numpy as np


def load_graphs(config, logger):
    # dataset_dir = args.data_dir+args.dataset+'/'
    dataset_dir = f'{config['data_dir']}/{config['dataset']}/'
    graphs = {}
    graphs['simplicies'] = read_simplicies(dataset_dir, mode='train')
    graphs['G_train'] = construct_graph(graphs['simplicies'])
    graphs['G_weighted'] = construct_weighted_graph(graphs['simplicies'])
    # graphs['simplicies_test'] = read_simplicies(dataset_dir,  mode='test')
    # graphs['G_test'] = construct_graph(graphs['simplicies_test'])
    logger.info('Finish loading graphs.')
    logger.info(f'Nodes train: {graphs['G_train'].number_of_nodes()}')
    logger.info(f'Simplicies train: {len(graphs['simplicies'])}')
    return graphs


def read_simplicies(file_dir, mode='train'):
    simplicies = []
    with open(file_dir + '{}.txt'.format(mode), 'r') as f:
        for line in f.readlines():
            simplicies.append(tuple(sorted(set([int(node) for node in line.strip().split(' ')]))))

    print(f'number of simplices: {len((simplicies))}')
    print(f'number of unique simplices: {len(set(simplicies))}')
    try:
        assert (len(set(simplicies)) == len(simplicies)) # no duplicate
        nodes = set([node for simplex in simplicies for node in simplex])
        assert(min(nodes) == 0)
        assert (max(nodes) == len(nodes)-1)  # compact indexing

    # sanity check
    except AssertionError:
        print('Node index should begin with 0, reindexing the hypergraphs ...')
        all_nodes = sorted(set(n for s in simplicies for n in s))
        node2i = {node: i for i, node in enumerate(all_nodes)}
        simplicies = [tuple(sorted(set([node2i[n] for n in s]))) for s in simplicies]

    return set(simplicies)


def construct_graph(simplicies):
    G = nx.Graph()
    for s in simplicies:
        if len(s) == 1:
            G.add_node(s[0])
            continue
        for e in it.combinations(s, 2):
            G.add_edge(*e)
    print('number of nodes in construct graph', G.number_of_nodes())
    return G


def construct_weighted_graph(simplicies):
    G = nx.Graph()
    for s in simplicies:
        if len(s) == 1:
            G.add_node(s[0])
            continue
        for u, v in it.combinations(s, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G


def construct_decomposed_graph(simplicies, level=2, limit=7):
    G = nx.Graph()
    for s in simplicies:
        if len(s) <= limit and len(s) >= level:
            hyper_simplicies = list(it.combinations(s, level))
            if len(hyper_simplicies) == 1:
                G.add_node(hyper_simplicies[0])
                continue
            for e in it.combinations(hyper_simplicies, 2):
                G.add_edge(*e)
    return G


def get_edges(clique):
    return it.combinations(clique, 2)


def get_clique_gain(weighted_adjacency_matrix, clique):
    gain = 0
    edges = get_edges(clique)
    for x, y in edges:
        if weighted_adjacency_matrix[x, y] > 0:
            gain += 2
        else:
            gain -= 2
    return gain


def lazy_clique_edge_cover(
    weighted_adjacency_matrix, clique_candidates, cliques_quota
):
    # build edge table
    edge_table = defaultdict(list)
    for idx, clique in enumerate(clique_candidates):
        edges = get_edges(clique)
        for edge in edges:
            edge_table[frozenset(edge)].append(idx)

    # default clique gain
    def get_default_gain(x):
        return len(x) * (len(x) - 1)

    clique_gain = np.array([get_default_gain(clique) for clique in clique_candidates])

    # the greedy process (import probability in here?)
    current_clique_idxs = []
    for i in range(cliques_quota):
        clique_gain[current_clique_idxs] = -10000
        best_clique_idx = np.argmax(clique_gain)
        current_clique_idxs.append(best_clique_idx)
        best_clique = clique_candidates[best_clique_idx]
        best_clique_edges = list(get_edges(best_clique))
        # update the weighted_adjacency_matrix
        to_update_edges = []
        for edge in best_clique_edges:
            x, y = edge
            weighted_adjacency_matrix[x, y] -= 1
            weighted_adjacency_matrix[y, x] -= 1
            if weighted_adjacency_matrix[x, y] == 0:
                to_update_edges.append(edge)

        # find the realted clique and update the clique gain
        for edge in to_update_edges:
            edge = frozenset(edge)
            cliques_idx = edge_table[edge]
            for idx in cliques_idx:
                clique_gain[idx] = clique_gain[idx] - 4
        

    return [clique_candidates[idx] for idx in current_clique_idxs]


def get_performance_wrt_ground_truth(reconstructed, ground_truth):
    correct_cliques = reconstructed & ground_truth
    precision = len(correct_cliques) / len(reconstructed) if len(reconstructed)>0 else 0
    recall = len(correct_cliques) / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall) if precision * recall > 0 else 0
    jaccard = len(correct_cliques) / len(reconstructed | ground_truth)
    return precision, recall, f1, jaccard