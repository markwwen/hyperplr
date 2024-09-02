import os
import time
import random
import yaml
import logging
from multiprocessing import Pool

import torch
import torch.nn as nn

from utils import load_graphs

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        # out: 1 x embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.literal_to_ix = literal_to_ix
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        # out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        # embeds = sum(self.embeddings(inputs)).view(1, -1)
        embeds = self.embeddings(inputs).sum(dim=0).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_vertex_embedding(self, vertex):
        ix = torch.tensor([vertex])
        return self.embeddings(ix)

    def get_embeddings(self):
        ix = torch.tensor([i for i in range(self.vocab_size)])
        return self.embeddings(ix)


# utils
def make_context_vector(context):
    return torch.tensor(context, dtype=torch.long)


def getEmbedding(simplicies, name):
    data = []
    vocab_size = 0
    for simplex in simplicies:
        simplex_len = len(simplex)
        if vocab_size < max(simplex):
            vocab_size = max(simplex)
        for i in range(simplex_len):
            context = [simplex[x] for x in range(simplex_len) if x != i]
            target = simplex[i]
            data.append((context, target))

    vocab_size += 1
    print(f"data size: {len(data)}")

    # model setting
    EMDEDDING_DIM = 50

    # literal_to_ix = {}
    # for i in range(1, num_vars + 1):
    #     literal_to_ix[i] = 2 * i - 2
    #     literal_to_ix[-i] = 2 * i - 1

    model = CBOW(vocab_size, EMDEDDING_DIM)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # training
    for epoch in range(50):
        total_loss = 0
        for context, target in data:
            context_vector = make_context_vector(context)
            log_probs = model(context_vector)
            total_loss += loss_function(
                log_probs, torch.tensor([target])
            )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, total_loss.item())

    # test the embedding
    embeddings = model.get_embeddings()
    torch.save(embeddings, f"./data/{name}/embedding.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    datasets = ['email-Eu', 'email-Enron', 'NDC-classes', 'contact-high-school', 'contact-primary-school']
    print(datasets)
    for name in datasets:
        print(name)
        config  = yaml.safe_load(open('./config.yml'))
        config['dataset'] = name
        config['beta'] = 150000
        graphs = load_graphs(config, logger)
        getEmbedding(graphs['simplicies_train'], name)

