import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """Two-layer fully connected model (MLP)."""

    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = x.to(dtype=self.fc1.weight.dtype, device=self.fc1.weight.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

    def output_num(self):
        return 1


class FuseModel(nn.Module):
    """Fusion model for vertical federated learning, accepts multiple party inputs."""

    def __init__(self, input_dim, output_dim, party_nums):
        super(FuseModel, self).__init__()
        self.party_nums = party_nums
        self.input_dim = input_dim
        self.concat_dim = input_dim * party_nums

        self.fuse_layer = nn.Linear(self.concat_dim, 64)
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, inputs):
        if self.party_nums > 1:
            x = [
                inp.to(
                    dtype=self.fuse_layer.weight.dtype, device=self.fuse_layer.weight.device
                )
                for inp in inputs
            ]
            x = torch.cat(x, dim=1)
        else:
            x = inputs
        x = F.relu(self.fuse_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
