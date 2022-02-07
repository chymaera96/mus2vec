import numpy as np
import torch
from .utils import *


del_p = 16
del_n_min = 1
del_n_max = 96

class SamplerMus2vec:

    def __init__(self, n_triplets, train, bias):

        self.n_triplets = n_triplets
        self.train = train
        self.bias = bias

    def compute_triplets(self, C_sync):
        bias = self.bias
        n_triplets = self.n_triplets
   
        if not bias:
            pass
        
        else:
            
            beat_segments = C_sync
            x_a = []
            x_p = []
            x_n = []
            for iter in range(n_triplets):

                i_a = np.random.choice(len(beat_segments))
                if i_a >= 20 and i_a <= len(beat_segments) - 21:
                    l1 = np.array(beat_segments[i_a - 8 : i_a])
                    l2 = np.array(beat_segments[i_a - 20 : i_a - 12])
                    cqt_l1 = np.transpose(l1,(0,2,1)).reshape(-1,l1.shape[1])
                    cqt_l2 = np.transpose(l2,(0,2,1)).reshape(-1,l2.shape[1])

                    r1 = np.array(beat_segments[i_a + 1 : i_a + 9])
                    r2 = np.array(beat_segments[i_a + 13 : i_a + 21])
                    cqt_r1 = np.transpose(r1,(0,2,1)).reshape(-1,r1.shape[1])
                    cqt_r2 = np.transpose(r2,(0,2,1)).reshape(-1,r2.shape[1])
                    
                    if distance(cqt_l1,cqt_l2) >= distance(cqt_r1,cqt_r2):
                        i_p = np.random.randint(i_a, min(i_a + del_p,len(beat_segments) - 1) + 1)
                        i_n = np.random.randint(max(i_a - del_n_max, 0), max(i_a - del_n_min, 0) + 1)
                    else:
                        i_p = np.random.randint(max(i_a - del_p, 0), i_a + 1)
                        i_n = np.random.randint(min(i_a + del_n_min, len(beat_segments) - 1), min(i_a + del_n_min, len(beat_segments) - 1) + 1)

                elif i_a < 20:
                    i_p = np.random.randint(i_a, min(i_a + del_p,len(beat_segments) - 1) + 1)
                    i_n = np.random.randint(np.random.randint(min(i_a + del_n_min, len(beat_segments) - 1), min(i_a + del_n_min, len(beat_segments) - 1) + 1))

                else:
                    i_p = np.random.randint(max(i_a - del_p, 0), i_a + 1)
                    i_n = np.random.randint(max(i_a - del_n_max, 0), max(i_a - del_n_min, 0) + 1)


                x_a.append(beat_segments[i_a])
                x_p.append(beat_segments[i_p])
                x_n.append(beat_segments[i_n])


        return torch.Tensor(x_a), torch.Tensor(x_p), torch.Tensor(x_n)


    def __call__(self, cqt_sync):

        x_a, x_p, x_n = self.compute_triplets(cqt_sync)

        return x_a, x_p, x_n

    