import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden=0, hidden_dim=0, activation=nn.Tanh()):
        super().__init__()
        assert num_hidden >= 0, "Number of hidden layers must be non-negative."
        assert hidden_dim > 0, "Number of hidden units must be positive."
        self.linear_layers = nn.ModuleList()
        if num_hidden == 0:
            self.linear_layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.linear_layers.append(nn.Linear(in_dim, hidden_dim))
            self.linear_layers.append(activation)
            for _ in range(num_hidden):
                self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.linear_layers.append(activation)
            self.linear_layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        return x


class Model:
    def __init__(self, obs_dim, act_dim):
        assert obs_dim > 0, "Observation dimension must be non-negative."
        assert act_dim > 0, "Action dimension must be non-negative."
        self._pi = MLP(in_dim=obs_dim, out_dim=act_dim, num_hidden=2, hidden_dim=50, activation=nn.ReLU())
        self._v = MLP(in_dim=obs_dim, out_dim=1, num_hidden=2, hidden_dim=50, activation=nn.ReLU())
    
    @property
    def pi(self):
        return self._pi

    @property
    def v(self):
        return self._v

    def compute_lprobs(self, obss, acts):
        dist = Categorical(self._pi(obss))
        lprobs = dist.log_prob(acts)
        return torch.sum(lprobs, dim=-1)

    def compute_acts(self, obss):
        dist = Categorical(logits=self._pi(obss))
        acts = dist.sample()
        return acts

    def compute_values(self, obss):
        return self._v(obss)
