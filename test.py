import yaml
import logging

import networkx as nx
from utils import load_graphs

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

config  = yaml.safe_load(open('./config.yaml'))
graph = load_graphs(config, logger)
G = graph['G_train']
print(nx.adjacency_matrix(G).todense())