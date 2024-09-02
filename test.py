import os
import time
from multiprocessing import Pool
import logging

import yaml
import networkx as nx
from utils import read_simplicies, construct_graph, construct_weighted_graph
from cliques import compute_cliques

PROCESS_NUM = 6

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def load_graphs(dataset_dir, logger):
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

# config  = yaml.safe_load(open('./config.yml'))


def load_maximal_clique(dataset, logger):
    dataset_dir = f'data/{dataset}/'
    graph = load_graphs(dataset_dir, logger)
    G = graph['G_train']
    start = time.time()
    cliques = nx.algorithms.clique.find_cliques(G)
    cliques = set(sorted([tuple(sorted(clique)) for clique in cliques]))
    # cliques = nx.algorithms.clique.enumerate_all_cliques(G)
    cliques = list(cliques)
    end = time.time()
    return f'{dataset}, {len(cliques)}, {end - start}' 


def print_callback(line):
    print(line)


if __name__ == '__main__':

    p = Pool(PROCESS_NUM)

    tasks = []
    datasets = os.listdir('./dataset')
    datasets = ['email-Eu',
 'email-Enron',
 'NDC-substances',
 'NDC-classes',
 'contact-high-school',
 'contact-primary-school']
    for dataset in datasets:
        task = p.apply_async(load_maximal_clique, args=(dataset, logger), callback=print_callback)
        tasks.append(task)

    tasks = [task.get() for task in tasks]
    print('tasks:', tasks)