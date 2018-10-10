import os
from os.path import join
import numpy as np
import torch
#from scipy.sparse import csr_matrix
import pandas as pd
import pickle
from torch.utils.data import Dataset

##########################################################################################################################

class TextDataset(Dataset):
    """datasets wrapper for cora, citeseer, pubmed, reddit"""

    def __init__(self, dataset_name, data_dir, subset='train'):
        """
        Args:
            data_dir (string): Directory for loading and saving train, test, and cv dataframes.
            download (boolean): Download newsgroups20 dataset from sklearn if necessary.
            subset (string): Specify subset of the datasets. The choices are: train, test, cv.
            bow_format (string): A weight scheme of a bag-of-words document. The choices are:
                tf (term frequency), tfidf (term freq with inverse document frequency), bm25.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.subset = subset
        fn = '{}.{}.pkl'.format(dataset_name, subset)
        self.df = self.load_df(data_dir, fn)

    def load_df(self, data_dir, df_file):
        df_file = os.path.join(data_dir, df_file)
        return pd.read_pickle(df_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        doc_bow = self.df.iloc[idx].bow
        doc_bow = torch.from_numpy(doc_bow.toarray().squeeze().astype(np.float32))
        label_bow = self.df.iloc[idx].label
        label_bow = torch.from_numpy(label_bow.toarray().squeeze().astype(np.float32))
        neighbors = self.df.iloc[idx].neighbors
        neighbors = torch.from_numpy(neighbors.toarray().squeeze().astype(np.float32))
        return (doc_bow, label_bow, neighbors)
    
    def num_classes(self):
        return self.df.iloc[0].label.shape[1]
    
##########################################################################################################################
