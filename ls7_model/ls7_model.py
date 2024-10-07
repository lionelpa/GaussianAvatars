from torch import nn

from scene import GaussianModel


class LS7Model(nn.Module):
    def __init__(self):
        super().__init__()
