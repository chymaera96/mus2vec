import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import warnings


min_len = 10
SAMPLE_RATE = 18000
n_segments = 64
songs_per_batch = 1

class Mus2vecDataset(Dataset):
    def __init__(self, path, json_dir, n_triplets=16, bias=True, train=True, sampler=None):
        # self.dataset = dataset
        self.path = path
        self.n_triplets = n_triplets
        self.bias = bias
        self.train = train
        self.sampler = sampler
        with open(json_dir) as json_file:
            self.filenames  = json.load(json_file)
        # self.input_shape = input_shape
        self.ignore_idx = []

    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = self.path + "/" + self.filenames[str(idx)]
        try:
            data = np.load(datapath)
        except Exception:
            print("Error loading:" + self.filenames[str(idx)])
            self.ignore_idx.append(idx)
            # self.filenames.pop(str(idx))
            return self[idx+1]
        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_a, x_p, x_n = self.sampler(data)

        return x_a, x_p, x_n
            

    def __len__(self):
        return len(self.filenames)