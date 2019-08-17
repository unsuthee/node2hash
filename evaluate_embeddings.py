import os
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch
import torch.optim as optim
from datasets import *
from utils import *
from model.EdgeReg import *
from model.EdgeReg_v2 import *
import argparse

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
parser.add_argument("-T", "--num_samples", default=1, type=int, help="number of samples from Q(z|x).")
parser.add_argument("--hash", action='store_true', help="enable this flag forces the model to hash the embedding before evaluation.")
parser.add_argument("--batch_size", default=100, type=int, help="the test batch size during evaluation.")

args = parser.parse_args()

if not args.gpunum:
    parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the number of bits.")

if args.hash:
    print("Evaluation on hash code.")
else:
    print("Evaluation on embedding vectors.")
    
##################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################################################

dataset_name = args.dataset
data_dir = os.path.join('dataset/clean', dataset_name)

train_batch_size = 100
test_batch_size = args.batch_size

train_set = TextDataset(dataset_name, data_dir, subset='train')
test_set = TextDataset(dataset_name, data_dir, subset='test')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=True)

#########################################################################################################
y_dim = train_set.num_classes()
num_bits = args.nbits
num_features = train_set[0][1].size(0)
num_nodes = len(train_set)
num_samples = args.num_samples

print("Train node2hash model ...")
print("dataset: {}".format(args.dataset))
print("numbits: {}".format(args.nbits))
print("T: {}".format(args.num_samples))
print("gpu id:  {}".format(args.gpunum))
print("num train: {} num test: {}".format(len(train_set), len(test_set)))

#########################################################################################################

if num_samples == 1:
    model = EdgeReg(dataset_name, num_features, num_nodes, num_bits, dropoutProb=0.1, device=device)
else:
    print("number of samples (T) = {}".format(num_samples))
    model = EdgeReg_v2(dataset_name, num_features, num_nodes, num_bits, dropoutProb=0.1, device=device, T=num_samples)

#########################################################################################################
if num_samples == 1:
    saved_model_file = 'saved_models/node2hash.{}.T{}.bit{}.pth'.format(dataset_name, num_samples, num_bits)
else:
    saved_model_file = 'saved_models/node2hash_v2.{}.T{}.bit{}.pth'.format(dataset_name, num_samples, num_bits)

print('load model {} ...'.format(saved_model_file))
model.load_state_dict(torch.load(saved_model_file))
model.to(device)
model.eval()

#########################################################################################################

import torch.nn.functional as F

if not args.hash:
    # get non-binary code
    with torch.no_grad():
        train_zy = [(model.encode(xb.to(model.device))[0], yb) for _, xb, yb, _ in train_loader]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)
        train_z_batch = train_z.unsqueeze(-1).transpose(2,0)

        prec_at_100 = []
        for _, xb, yb, _ in tqdm(test_loader, ncols=80):
            test_z = model.encode(xb.to(model.device))[0]
            test_y = yb
            test_z_batch = test_z.unsqueeze(-1)

            # compute cosine similarity
            dist = F.cosine_similarity(test_z_batch, train_z_batch, dim=1)
            ranklist = torch.argsort(dist, dim=1, descending=True)
            top100 = ranklist[:, :100]

            for eval_index in range(0, test_y.size(0)):
                top100_labels = torch.index_select(train_y.to(device), 0, top100[eval_index]).type(torch.cuda.ByteTensor)
                groundtruth_label = test_y[eval_index].type(torch.cuda.ByteTensor)
                matches = (groundtruth_label.unsqueeze(0) & top100_labels).sum(dim=1) > 0
                num_corrects = matches.sum().type(torch.cuda.FloatTensor)
                prec_at_100.append((num_corrects/100.).item())   

        avg_prec_at_100 = np.mean(prec_at_100)
        print('Nonhash: average prec at 100 = {:.4f}'.format(avg_prec_at_100))

        with open('nonbinary_logs/Nonbinary.Experiment.{}.txt'.format(args.dataset), 'a') as handle:
            handle.write('{}\t{}\t{}\t{}\n'.format(args.dataset, args.nbits, args.num_samples, avg_prec_at_100))
            
else:
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        avg_prec_at_100 = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100)
        print('Hash: average prec at 100 = {:.4f}'.format(avg_prec_at_100))
        
        with open('binary_logs/binary.Experiment.{}.txt'.format(args.dataset), 'a') as handle:
            handle.write('{}\t{}\t{}\t{}\n'.format(args.dataset, args.nbits, args.num_samples, avg_prec_at_100))