import argparse
import importlib
import json
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")


def main(input_json):
    # Load the input image from the JSON file
    with open(input_json, 'r') as f:
        input_data = json.load(f)
        input_image = Image.open(input_data['image'])
        checkpoint = input_data['checkpoint']
        model_name = input_data['model_name']

    # Load the PyTorch model
    module = importlib.import_module("models." + model_name)
    model = module.get_model()
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Apply the transform to the input image and add a batch dimension
    input_tensor = module.pre_processing(input_image).to(device)

    # Make a prediction on the input image using the PyTorch model
    with torch.no_grad():
        output_tensor = model(input_tensor)

    module.post_processing(output_tensor)
    print(output_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonfile', type=str, help='Input file')
    args = parser.parse_args()
    main(args.jsonfile)
