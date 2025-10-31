import os

import torch
from dotenv import load_dotenv

from src.u2net import U2NET


load_dotenv()


config = {
    "in_ch": 3,
    "out_ch": 1
}

state_dict = torch.load("checkpoints/u2net.pth", map_location="cpu")

model = U2NET(**config)
model.load_state_dict(
    {
        f"model.{k}": v for k, v in state_dict.items()
    }
)

model.save_pretrained("U2NET")
model.push_to_hub("papermill/U2NET", token=os.getenv("HF_HUB_TOKEN"))
