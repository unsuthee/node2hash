import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def retrieve_topk(query_b, doc_b, topK, batch_size=100):
    n_bits = doc_b.size(1)
    n_train = doc_b.size(0)
    n_test = query_b.size(0)

    topScores = torch.cuda.ByteTensor(n_test, topK + batch_size).fill_(n_bits+1)
    topIndices = torch.cuda.LongTensor(n_test, topK + batch_size).zero_()

    testBinmat = query_b.unsqueeze(2)

    for batchIdx in tqdm(range(0, n_train, batch_size), ncols=0, leave=False):
        s_idx = batchIdx
        e_idx = min(batchIdx + batch_size, n_train)
        numCandidates = e_idx - s_idx

        trainBinmat = doc_b[s_idx:e_idx]
        trainBinmat.unsqueeze_(0)
        trainBinmat = trainBinmat.permute(0, 2, 1)
        trainBinmat = trainBinmat.expand(testBinmat.size(0), n_bits, trainBinmat.size(2))

        testBinmatExpand = testBinmat.expand_as(trainBinmat)

        scores = (trainBinmat ^ testBinmatExpand).sum(dim=1)
        indices = torch.arange(start=s_idx, end=e_idx, step=1).type(torch.cuda.LongTensor).unsqueeze(0).expand(n_test, numCandidates)

        topScores[:, -numCandidates:] = scores
        topIndices[:, -numCandidates:] = indices

        topScores, newIndices = topScores.sort(dim=1)
        topIndices = torch.gather(topIndices, 1, newIndices)

    return topIndices

def compute_precision_at_k(retrieved_indices, query_labels, doc_labels, topK):
    n_test = query_labels.size(0)
    
    Indices = retrieved_indices[:,:topK]
    
    topTrainLabels = [torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0) for idx in range(0, n_test)]
    topTrainLabels = torch.cat(topTrainLabels, dim=0).type(torch.cuda.ShortTensor)
    test_labels = query_labels.unsqueeze(1).expand(n_test, topK, topTrainLabels.size(-1)).type(torch.cuda.ShortTensor)
    relevances = (topTrainLabels & test_labels).sum(dim=2)
    relevances = (relevances > 0).type(torch.cuda.ShortTensor)
        
    true_positive = relevances.sum(dim=1).type(torch.cuda.FloatTensor)
    true_positive = true_positive.div_(100)
    prec_at_k = torch.mean(true_positive)
    return prec_at_k

def Random_walk(df, start_node_id, num_steps):
    visited_nodes = set()
    step_count = 0
    curr_node_id = start_node_id
    visited_nodes = set()
    visited_nodes.add(curr_node_id)
    while step_count < num_steps:
        nn_list = df.loc[curr_node_id].neighbors.nonzero()[1]
        curr_node_id = np.random.choice(nn_list)
        visited_nodes.add(int(curr_node_id))
        step_count += 1
    #visited_nodes.remove(start_node_id)
    return list(visited_nodes)

def BFS_walk(df, start_node_id, num_steps):
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
        
        nn_list = list(df.loc[curr_node_id].neighbors.nonzero()[1])
        np.random.shuffle(nn_list)
        queue += nn_list
        visited_nodes.add(curr_node_id)
        curr_step += 1
        if curr_step > num_steps:
            break
    
    #if not isinstance(start_node_id, list):
    #    visited_nodes.remove(start_node_id)
    return list(visited_nodes)    

def DFS_walk(df, start_node_id, num_hops):
    if isinstance(start_node_id, list):
        stack = list(start_node_id)
    else:
        stack = [start_node_id]
        
    visited_nodes = set()
    curr_hop = 0
    while len(stack) > 0:         
        curr_node_id = stack.pop()
        while curr_node_id in visited_nodes:
            if len(stack) <= 0:
                #if not isinstance(start_node_id, list):
                #    visited_nodes.remove(start_node_id)
                return list(visited_nodes)
            curr_node_id = stack.pop()
        
        nn_list = list(df.loc[curr_node_id].neighbors.nonzero()[1])
        np.random.shuffle(nn_list)
        stack += nn_list
        visited_nodes.add(curr_node_id)
        curr_hop += 1
        if curr_hop > num_hops:
            break
            
    #if not isinstance(start_node_id, list):
    #    visited_nodes.remove(start_node_id)
    return list(visited_nodes)