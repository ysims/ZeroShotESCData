import torch
from torch import nn
import models.utils as utils


class Diffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, aux_dim):
        super(Diffusion, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim + aux_dim, hidden_dim),
            # nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

        # init weights
        # utils.weights_init_xavier_uniform(self.l1)
        # utils.weights_init_xavier_uniform(self.l2)

    def forward(self, x, aux):
        # Ensure x and aux are 2D tensors
        x = x if x.dim() > 1 else x.unsqueeze(0)
        aux = aux if aux.dim() > 1 else aux.unsqueeze(0)

        x = torch.cat((x, aux), dim=1)  # concatenate x and aux
        x = self.layers(x)
        return x

    # Distort the input data by adding Gaussian noise
    def distort(self, x, epoch_percentage):
        noise_std = 0.1 * epoch_percentage  # Scale the noise standard deviation with epoch percentage
        noise = torch.randn_like(x) * noise_std
        return x + noise