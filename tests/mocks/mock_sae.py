from submodules.SAeUron.SAE.sae import Sae
import torch


class MockSae(Sae):
    def __init__(self, input_dim, hidden_dim):
        self.W_enc = torch.eye(input_dim, hidden_dim)
        self.W_dec = torch.eye(hidden_dim, input_dim)
        self.b_dec = torch.zeros(input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def preprocess_input(self, x):
        return x, None, None

    def pre_acts(self, x):
        return x @ self.W_enc

    def select_topk(self, latents, k=None, batch_size=None):
        if k is None:
            k = self.hidden_dim
        return torch.topk(latents, k=k)
