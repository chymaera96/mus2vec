import torch
import torch.nn as nn

class BatchTripletLoss(nn.Module):
    def __init__(self, batch_size, n_triplets, margin):
        super(BatchTripletLoss, self).__init__()
        self.batch_size = batch_size
        self.n_triplets = n_triplets
        self.margin = margin

        self.criterion = nn.TripletMarginLoss(margin=self.margin, reduction="sum")
    

    def forward(self, z_a, z_p, z_n):

        N = self.n_triplets * self.batch_size
        loss = self.criterion(z_a, z_p, z_n)
        loss /= N

        return loss