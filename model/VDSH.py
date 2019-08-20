import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class VDSH(nn.Module):
    
    def __init__(self, dataset, vocabSize, numNodes, 
                       latentDim, device, dropoutProb=0., T=1):
        super(VDSH, self).__init__()
        
        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.numNodes = numNodes
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.device = device
        self.T = T # number of samples from Q(z|x)
        
        self.encoder = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb))
        
        self.h_to_mu = nn.Linear(self.hidden_dim, self.latentDim)
        self.h_to_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
                                         nn.Sigmoid())
        
        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
                                     nn.LogSoftmax(dim=1))
        
    def use_content(self):
        return True
    
    def use_neighbors(self):
        return False
    
    def encode(self, doc_mat):
        h = self.encoder(doc_mat)
        z_mu = self.h_to_mu(h)
        z_logvar = self.h_to_logvar(h)
        return z_mu, z_logvar
        
    def reparametrize(self, mu, logvar):
        eps = mu.new_empty((mu.size(0), self.T, mu.size(1))).normal_()
        std = torch.sqrt(torch.exp(logvar)).unsqueeze(1)
        z = eps.mul(std).add_(mu.unsqueeze(1))
        return z
    
    def forward(self, document_mat):
        mu, logvar = self.encode(document_mat)
        # z is (# batch, T, #latent_dim)
        z = self.reparametrize(mu, logvar) 
        
        log_prob_w = self.decoder(z.view(-1, self.latentDim))
        log_prob_w = log_prob_w.view(mu.size(0), self.T, -1)
        return log_prob_w, mu, logvar
    
    def get_name(self):
        return "VDSH"
    
    @staticmethod
    def calculate_KL_loss(mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD).mul_(-0.5)
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        '''
        Return a float value 
        '''
        return -torch.mean(torch.sum(logprob_word * doc_mat.unsqueeze(1), dim=2))
    
    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.to(self.device))[0], yb) for _, xb, yb, _ in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device))[0], yb) for _, xb, yb, _ in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        mid_val, _ = torch.median(train_z, dim=0)
        train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
        test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y
        