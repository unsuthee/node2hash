import os
import numpy as np
import pandas as pd
import scipy.io
from scipy import sparse
import pickle
from tqdm import tqdm

##########################################################################################
dataset_name = 'reddit'
data_path = os.path.join('../dataset/clean/reddit')

train_graph = pickle.load(open(os.path.join(data_path, "ind.reddit.train.graph.pkl"), "rb"))
test_graph = pickle.load(open(os.path.join(data_path, "ind.reddit.test.graph.pkl"), "rb"))

data = scipy.io.loadmat(os.path.join(data_path, 'ind.reddit.mat'))
train_features = data['train']
test_features = data['test']
gnd_train = data['gnd_train']
gnd_test = data['gnd_test']

# create a connection matrix
n_train = train_features.shape[0]
row = []
col = []
for doc_id in tqdm(train_graph):
    row += [doc_id] * len(train_graph[doc_id])
    col += train_graph[doc_id]
data = [1] * len(row)
train_connections = sparse.csr_matrix((data, (row, col)), shape=(n_train, n_train))

n_test = test_features.shape[0]
row = []
col = []
for doc_id in tqdm(test_graph):
    row += [doc_id] * len(test_graph[doc_id])
    col += test_graph[doc_id]
data = [1] * len(row)
test_connections = sparse.csr_matrix((data, (row, col)), shape=(n_test, n_train)) # test graph points to train graph


save_dir = os.path.join('../dataset/clean', dataset_name)
##########################################################################################

train = []
for doc_id in tqdm(train_graph):
    doc = {'doc_id': doc_id, 'bow': train_features[doc_id], 
           'label': gnd_train[doc_id], 'neighbors': train_connections[doc_id]}
    train.append(doc)

train_df = pd.DataFrame.from_dict(train)
train_df.set_index('doc_id', inplace=True)

fn = os.path.join(save_dir, '{}.train.pkl'.format(dataset_name))
train_df.to_pickle(fn)
##########################################################################################

test = []
for doc_id in tqdm(test_graph):
    doc = {'doc_id': doc_id, 'bow': test_features[doc_id], 
           'label': gnd_test[doc_id], 'neighbors': test_connections[doc_id]}
    test.append(doc)

test_df = pd.DataFrame.from_dict(test)
test_df.set_index('doc_id', inplace=True)

fn = os.path.join(save_dir, '{}.test.pkl'.format(dataset_name))
test_df.to_pickle(fn)