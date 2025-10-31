import os

from dotenv import load_dotenv

from src.u2net import U2NET


load_dotenv()


config = {
    "in_ch": 3,
    "out_ch": 1
}

model = U2NET(**config)

model.save_pretrained("U2NET")
model.push_to_hub("papermill/U2NET", token=os.getenv("HF_HUB_TOKEN"))
