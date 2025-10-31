import os

from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from dotenv import load_dotenv

from src.u2net import U2NET as BaseModel


load_dotenv()


class U2NET(nn.Module, PyTorchModelHubMixin):
    def __init__(self, in_ch: int, out_ch: int):
        super(U2NET, self).__init__()

        self.model = BaseModel(in_ch, out_ch)

    def forward(self, x):
        self.model(x)


config = {
    "in_ch": 3,
    "out_ch": 1
}

model = U2NET(**config)

model.save_pretrained("U2NET")
model.push_to_hub("papermill/U2NET", token=os.getenv("HF_HUB_TOKEN"))
