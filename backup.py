from functools import reduce
import operator as op
import random
from math import comb
from collections import defaultdict, Counter
import networkx as nx


# import numpy as np
# import seaborn as sns
# # import matplotlib.pyplot as plt

# data = np.array(clique_sizes)
# print(data)
# # hist, bins = np.histogram(data, bins=np.linspace(0, 25, 26))
# # print(hist)
# # print(bins)
# sns.displot(data, bins=range(30))

# for clique in cliques:
#     if len(clique) <= 10:
#         if len(clique) >= 3:
#             clique_candidates.append(clique)
#     else:
#         break

# print(clique_candidates[9])
# print(len(clique_candidates))
# print(maximum_cliques)


# def sample_random(cliques, beta):
#     # cal the weight of clique by size
#     weights = [min(2**len(c), 2**30) for c in cliques]
#     clique_indicies = random.choices(range(len(cliques)), weights=weights, k=beta)
#     children = defaultdict(list)
#     for i in clique_indicies:
#         clique = cliques[i]
#         n = len(clique)
#         subset_weights = [comb(n, r) for r in range(1, n+1)]
#         subset_size = random.choices(range(1, n+1), weights=subset_weights, k=1)[0]
#         children[clique].append(tuple(sorted(random.sample(clique, subset_size))))
#     return children



# graph_path = './dataset/coauth-DBLP-proj-graph/coauth-DBLP-proj-graph.txt'
# graph_path = './dataset/contact-high-school-proj-graph/contact-high-school-proj-graph.txt'

# G = nx.read_weighted_edgelist(graph_path)
# print(f'Number of nodes: {G.number_of_nodes()}')
# maximum_cliques = [c for c in nx.find_cliques(G)]
# print(len(maximum_cliques))
# maximum_cliques = [tuple(sorted(clique)) for clique in maximum_cliques]
# clique_sizes = [len(c) for c in maximum_cliques]
# print(max(clique_sizes), min(clique_sizes))
# A = nx.adjacency_matrix(G)
# print(A.todense().shape)