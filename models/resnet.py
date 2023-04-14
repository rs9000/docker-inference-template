import json

from torchvision import models, transforms


def get_model():
    model = models.resnet18(pretrained=False, num_classes=1000)
    return model


def pre_processing(input_image):
    # Define the transform to apply to the input image
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transform to the input image and add a batch dimension
    return transform(input_image).unsqueeze(0)


def post_processing(output_tensor):
    # Convert the tensor to a JSON-serializable dictionary
    tensor_dict = {'data': output_tensor.tolist()}

    # Save it to file
    with open('json/results.json', 'w') as f:
        json.dump(tensor_dict, f)
