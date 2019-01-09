import os
from os.path import join
import numpy as np
import pandas as pd
from scipy import sparse
import pickle
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from utils import *
from tqdm import tqdm
from model.EdgeReg import *

#######################################################################################################
class TextAndNearestNeighborsDataset(Dataset):

    def __init__(self, dataset_name, data_dir, subset='train'):
        """
        Args:
            data_dir (string): Directory for loading and saving train, test, and cv dataframes.
            subset (string): Specify subset of the datasets. The choices are: train, test, cv.
        """
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(data_dir, dataset_name)
        self.subset = subset
        fn = '{}.{}.pkl'.format(dataset_name, subset)
        self.df = self.load_df(self.data_dir, fn)
        self.docid2index = {docid: index for index, docid in enumerate(list(self.df.index))}
        
        if dataset_name in ['reuters', 'rcv1', 'tmc']:
            self.single_label = False
        else:
            self.single_label = True

    def load_df(self, data_dir, df_file):
        df_file = os.path.join(data_dir, df_file)
        return pd.read_pickle(df_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        doc_id = self.df.iloc[idx].name
        doc_bow = self.df.iloc[idx].bow
        doc_bow = torch.from_numpy(doc_bow.toarray().squeeze().astype(np.float32))
        
        label = self.df.iloc[idx].label
        label = torch.from_numpy(label.toarray().squeeze().astype(np.float32))
                
        neighbors = torch.LongTensor(self.df.iloc[idx].neighbors)
        return (doc_id, doc_bow, label, neighbors)
    
    def num_classes(self):
        return self.df.iloc[0].label.shape[1]
    
    def num_features(self):
        return self.df.iloc[0].bow.shape[1]
    
#######################################################################################################
train_set = TextAndNearestNeighborsDataset('ng20', 'dataset/clean', 'train')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)
test_set = TextAndNearestNeighborsDataset('ng20', 'dataset/clean', 'test')
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=100, shuffle=True)

#######################################################################################################
def BFS_walk(df, start_node_id, num_steps, max_branch_factor=20):
    if isinstance(start_node_id, list):
        queue = list(start_node_id)
    else:
        queue = [start_node_id]
        
    visited_nodes = set()
    curr_step = 0
    while len(queue) > 0:         
        curr_node_id = queue.pop(0)
        while curr_node_id in visited_nodes:
            if len(queue) <= 0:
                #if not isinstance(start_node_id, list):
                #    visited_nodes.remove(start_node_id)
                return list(visited_nodes)
            curr_node_id = queue.pop(0)
        
        nn_list = list(train_set.df.loc[curr_node_id].neighbors[:max_branch_factor])
        #np.random.shuffle(nn_list)
        queue += nn_list
        visited_nodes.add(curr_node_id)
        curr_step += 1
        if curr_step > num_steps:
            break
    
    #if not isinstance(start_node_id, list):
    #    visited_nodes.remove(start_node_id)
    return list(visited_nodes)    

#######################################################################################################
walk_type, max_nodes = 'BFS', 20
max_nodes = int(max_nodes)
print("Walk type: {} with maximum nodes of: {}".format(walk_type, max_nodes))

if walk_type == 'BFS':
    neighbor_sample_func = BFS_walk
elif walk_type == 'DFS':
    neighbor_sample_func = DFS_walk
elif walk_type == 'Random':
    neighbor_sample_func = Random_walk
else:
    neighbor_sample_func = None
    print("The model will only takes the immediate neighbors.")
    #assert(False), "unknown walk type (has to be one of the following: BFS, DFS, Random)"

def get_neighbors(ids, df, max_nodes, batch_size, traversal_func):
    cols = []
    rows = []
    for idx, node_id in enumerate(ids):
        nn_indices = traversal_func(df, node_id.item(), max_nodes)
        col = [train_set.docid2index[v] for v in nn_indices]
        rows += [idx] * len(col)
        cols += col
    data = [1] * len(cols)
    connections = sparse.csr_matrix((data, (rows, cols)), shape=(batch_size, len(df)))
    return torch.from_numpy(connections.toarray()).type(torch.FloatTensor)

#######################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######################################################################################################

dataset_name = 'ng20'
y_dim = train_set.num_classes()
num_bits = 32
num_features = train_set[0][1].size(0)
num_nodes = len(train_set)
edge_weight = 1.0

#######################################################################################################
model = EdgeReg(dataset_name, num_features, num_nodes, num_bits, dropoutProb=0.1, device=device)
model.to(device)

#######################################################################################################

optimizer = optim.Adam(model.parameters(), lr=0.01)
kl_weight = 0.
kl_step = 1 / 5000.

edge_weight = 0.
edge_step = 1 / 1000.

best_precision = 0
best_precision_epoch = 0

for epoch in range(20):
    avg_loss = []
    for ids, xb, yb, nb in tqdm(train_loader, ncols=50):
        xb = xb.to(device)
        yb = yb.to(device)

        nb = get_neighbors(ids, train_set.df, max_nodes, xb.size(0), neighbor_sample_func)
        nb = nb.to(device)

        logprob_w, logprob_nn, mu, logvar = model(xb)
        kl_loss = EdgeReg.calculate_KL_loss(mu, logvar)
        reconstr_loss = EdgeReg.compute_reconstr_loss(logprob_w, xb)
        nn_reconstr_loss = EdgeReg.compute_edge_reconstr_loss(logprob_nn, nb)

        loss = reconstr_loss + edge_weight * nn_reconstr_loss + kl_weight * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kl_weight = min(kl_weight + kl_step, 1.)
        edge_weight = min(edge_weight + edge_step, 1.)

        avg_loss.append(loss.item())
        
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100)
        print("precision at 100: {:.4f}".format(prec.item()))

        if prec.item() > best_precision:
            best_precision = prec.item()
            best_precision_epoch = epoch + 1

        print('{} epoch:{} loss:{:.4f} Best Precision:({}){:.3f}'.format(model.get_name(), epoch+1, np.mean(avg_loss), best_precision_epoch, best_precision))

        