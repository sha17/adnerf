import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

from importlib import import_module

def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

class PositionEncoder(nn.Module):
    def __init__(
        self,
        input_dims=3,
        include_input=True,
        max_freq_log2=9,
        num_freqs=10,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos]
    ):
        super().__init__()

        embed_fns = []

        out_dim = 0
        if include_input:
            embed_fns.append(lambda x: x)
            out_dim += input_dims

        max_freq = num_freqs-1
        N_freqs = num_freqs

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        periodic_fns = [load_func(p_fn) for p_fn in periodic_fns]

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += input_dims

        self.out_dim = out_dim
        self.embed_fns = embed_fns

    def get_output_dim(self,):
        return self.out_dim

    def forward(self, x):
        x = torch.cat([fn(x) for fn in self.embed_fns], -1)
        return x