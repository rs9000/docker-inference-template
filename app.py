import importlib
from typing import Any, Dict

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from model_registry import model_registry

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

app = FastAPI()
origins = ["localhost"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DynamicItem(BaseModel):
    data: Dict[str, Any]


@app.post("/predict/")
async def predict(input_json: DynamicItem):
    print(input_json.data)
    if "checkpoint" in input_json.data.keys():
        checkpoint = input_json.data["checkpoint"]
    else:
        checkpoint = None

    model_name = input_json.data["model_name"]

    # Load the Models in the zoo
    module = importlib.import_module("models." + model_name)
    class_name = model_registry.get(model_name)

    # Check if the model class name exists in the registry
    if class_name:
        # Instantiate the model
        model_class = getattr(module, class_name)
        model = model_class(input_json)
        model.load_model()
    else:
        print("Model not found in the registry.")

    # Load the model weights
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.model.load_state_dict(checkpoint)

    # Make a prediction on the input image using the PyTorch model
    with torch.no_grad():
        json_out = model.predict()

    print(json_out)
    return json_out


if __name__ == "__main__":
    # Tests
    predict(input_json=DynamicItem(data={"image": "./images/sample.jpg",
                                         "checkpoint": "./checkpoints/resnet18-5c106cde.pth",
                                         "model_name": "resnet18"}))

    predict(input_json=DynamicItem(data={"image": "./images/sample.jpg",
                                         "checkpoint": "./checkpoints/efficientnet_b0_rwightman-3dd342df.pth",
                                         "model_name": "efficientnet_b0"}))

    predict(input_json=DynamicItem(
        data={"positive_prompt": "An image of an helicopter flying in front of a mountain, high quality",
              "negative_prompt": "low quality",
              "model_name": "stablediffusion2",
              "save_path": "./df.png"}))
