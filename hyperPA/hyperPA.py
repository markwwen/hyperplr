import itertools as it
from collections import defaultdict, Counter

import networkx as nx



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


def compute_np_distribution(file_path):
    simplicies = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            simplicies.append(tuple(sorted(set([int(node) for node in line.strip().split(' ')]))))

    print(f'number of simplices: {len((simplicies))}')
    print(f'number of unique simplices: {len(set(simplicies))}')

    # reindex
    all_nodes = list(dict.fromkeys([n for s in simplicies for n in s]).keys())
    node2i = {node: i for i, node in enumerate(all_nodes)}
    reindex_simplicies = []
    for s in simplicies:
        s_prime = tuple(node2i[x] for x in s)
        reindex_simplicies.append(s_prime)

    # get HE
    nodes_num = len(all_nodes)
    HE = [0] * nodes_num
    for s in reindex_simplicies:
        for i in range(max(s), nodes_num):
            HE[i] += 1

    # get np
    np = []
    for i in range(nodes_num - 1):
        np.append(HE[i+1] - HE[i])

    np_distribution = Counter(np)
    print(np_distribution)

    return np_distribution


compute_np_distribution('./email-Eu.txt')