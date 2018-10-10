import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from datasets import *
from utils import *
from model.EdgeReg import *
import argparse

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
parser.add_argument("--dropout", help="Dropout probability (0 means no dropout)", default=0.1, type=float)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--transform_batch_size", default=100, type=int)
parser.add_argument("--num_epochs", default=30, type=int)
parser.add_argument("--lr", default=0.001, type=float)

args = parser.parse_args()

if not args.gpunum:
    parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the number of bits.")
        
##################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################################################

dataset_name = args.dataset
data_dir = os.path.join('dataset/clean', dataset_name)

train_set = TextDataset(dataset_name, data_dir, subset='train')
test_set = TextDataset(dataset_name, data_dir, subset='test')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=True)

#########################################################################################################
y_dim = train_set.num_classes()
num_bits = args.nbits
num_features = train_set[0][0].size(0)
num_nodes = len(train_set)

print("Train node2hash model ...")
print("dataset: {}".format(args.dataset))
print("numbits: {}".format(args.nbits))
print("gpu id:  {}".format(args.gpunum))
print("dropout probability: {}".format(args.dropout))
print("num epochs: {}".format(args.num_epochs))
print("learning rate: {}".format(args.lr))
print("num train: {} num test: {}".format(len(train_set), len(test_set)))

#########################################################################################################

model = EdgeReg(dataset_name, num_features, num_nodes, num_bits, dropoutProb=0.1, device=device)
model.to(device)

num_epochs = args.num_epochs

optimizer = optim.Adam(model.parameters(), lr=args.lr)
kl_weight = 0.
kl_step = 1 / 5000.

best_precision = 0
best_precision_epoch = 0

with open('logs/EdgeReg/loss.log.txt', 'w') as log_handle:
    log_handle.write('epoch,step,loss,reconstr_loss,nn_reconstr_loss,kl_loss\n')
    
    for epoch in range(num_epochs):
        avg_loss = []
        for step, (xb, yb, nb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            nb = nb.to(device)
            
            logprob_w, logprob_nn, mu, logvar = model(xb)
            kl_loss = EdgeReg.calculate_KL_loss(mu, logvar)
            reconstr_loss = EdgeReg.compute_reconstr_loss(logprob_w, xb)
            nn_reconstr_loss = EdgeReg.compute_edge_reconstr_loss(logprob_nn, nb)

            loss = reconstr_loss + nn_reconstr_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            avg_loss.append(loss.item())
            
            log_handle.write('{},{},{:.4f},{:.4f},{:.4f}'.format(epoch, step, loss.item(), 
                                                                 reconstr_loss.item(), nn_reconstr_loss.item(), kl_loss.item()))
        print('{} epoch:{} loss:{:.4f} Best Precision:({}){:.3f}'.format(model.get_name(), epoch+1, np.mean(avg_loss), best_precision_epoch, best_precision))
        
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100)
            print("precision at 100: {:.4f}".format(prec.item()))

            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
        
#########################################################################################################
with open('logs/EdgeReg/result.txt', 'a') as handle:
    handle.write('{},{},{},{}\n'.format(dataset_name, args.nbits, best_precision_epoch, best_precision))