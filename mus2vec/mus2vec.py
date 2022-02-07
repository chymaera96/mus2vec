import torch.nn as nn
import torch.nn.functional as F

class Mus2vec(nn.Module):

    def __init__(self, encoder, input_dim = 1536, projection_dim=128, n_features=128):
        super(Mus2vec, self).__init__()

        self.encoder = encoder
        self.input_dim = input_dim
        self.n_features = n_features
        self.projection_dim = projection_dim

        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, self.projection_dim),
        )


    def forward(self, x_a, x_p, x_n):
        h_a = self.encoder(x_a)
        z_a = self.projector(h_a)
        z_a = F.normalize(z_a, p=2)
        
        h_p = self.encoder(x_p)
        z_p = self.projector(h_p)
        z_p = F.normalize(z_p, p=2)

        h_n = self.encoder(x_n)
        z_n = self.projector(h_n)
        z_n = F.normalize(z_n, p=2)


        return z_a, z_p, z_n