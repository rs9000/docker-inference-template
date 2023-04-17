import importlib
import torch
from fastapi import FastAPI
from pydantic import BaseModel

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

app = FastAPI()


class Item(BaseModel):
    image: str
    checkpoint: str
    model_name: str


@app.post("/predict/")
async def predict(input_json: Item):
    input_data = input_json
    print(input_data)

    checkpoint = input_data.checkpoint
    model_name = input_data.model_name

    # Load the PyTorch model
    module = importlib.import_module("models." + model_name)
    model = module.get_model()
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Apply the transform to the input image and add a batch dimension
    input_tensor = module.pre_processing(input_data).to(device)

    # Make a prediction on the input image using the PyTorch model
    with torch.no_grad():
        output_tensor = model(input_tensor)

    json_out = module.post_processing(output_tensor)
    return json_out
