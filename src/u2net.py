from torch import nn
from huggingface_hub import PyTorchModelHubMixin

from models.u2net import U2NET as U2NETBase


class U2NET(nn.Module, PyTorchModelHubMixin):
    def __init__(self, in_ch: int, out_ch: int):
        super(U2NET, self).__init__()

        self.model = U2NETBase(in_ch, out_ch)

    def forward(self, x):
        return self.model(x)
