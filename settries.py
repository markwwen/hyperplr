import os
import time
import pickle

import networkx as nx

from utils import read_simplicies, construct_graph, construct_weighted_graph
from cliques import compute_cliques

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_set = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, s):
        node = self.root
        for element in sorted(s):  # Ensure the set elements are in a consistent order
            if element not in node.children:
                node.children[element] = TrieNode()
            node = node.children[element]
        node.is_end_of_set = True

    def find_all_supersets(self, s):
        def dfs(node, path, results):
            if node.is_end_of_set and set(path).issuperset(s):
                results.append(set(path))  # Found a superset, add to results
            
            for element in node.children:
                dfs(node.children[element], path + [element], results)

        results = []
        dfs(self.root, [], results)
        return results


def load_graphs(dataset_dir):
    # dataset_dir = args.data_dir+args.dataset+'/'
    # dataset_dir = f'{config['data_dir']}/{config['dataset']}/'
    graphs = {}
    graphs['simplicies_train'] = read_simplicies(dataset_dir, mode='train')
    graphs['G_train'] = construct_graph(graphs['simplicies_train'])
    graphs['G_weighted'] = construct_weighted_graph(graphs['simplicies_train'])
    # graphs['simplicies_test'] = read_simplicies(dataset_dir,  mode='test')
    # graphs['G_test'] = construct_graph(graphs['simplicies_test'])
    # logger.info('Finish loading graphs.')
    # logger.info(f'Nodes train: {graphs['G_train'].number_of_nodes()}')
    # logger.info(f'Simplicies train: {len(graphs['simplicies_train'])}')
    return graphs

# # Example usage
# trie = Trie()
# sets_list = [{1, 2, 3}, {1, 3}, {1, 2}]
# for s in sets_list:
#     trie.insert(s)

# input_set = {1}
# supersets = trie.find_all_supersets(input_set)
# print(supersets)

# dataset loadding
dataset = 'NDC-substances'
dataset_dir = f'data/{dataset}/'
start = time.time()
cache = f'data/{dataset}/cliques.pkl'
if os.path.exists(cache):
    print(f'Found cache for max cliques')
    with open(cache, 'rb') as c:
        max_cliques = pickle.load(c)
else:
    graph = load_graphs(dataset_dir)
    G = graph['G_train']
    cliques = nx.algorithms.clique.find_cliques(G)
    max_cliques = set(sorted([tuple(sorted(clique)) for clique in cliques]))
    with open(cache, 'wb') as c:
        pickle.dump(max_cliques, c)
end = time.time()
print(f'get maximal cliques: {dataset}, {len(max_cliques)}, {end - start}')

# build set-tries
start = time.time()
max_cliques = [set(x) for x in max_cliques]
trie = Trie()
for clique in max_cliques:
    trie.insert(clique)
end = time.time()
print(f'build set tries: {dataset}, {len(max_cliques)}, {end - start}')
