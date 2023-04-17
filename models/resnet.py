from PIL import Image
from torchvision import models, transforms


def get_model():
    """
    Load the model
    :return: model
    """
    model = models.resnet18(pretrained=False, num_classes=1000)
    return model


def pre_processing(input_json):
    """
    Take the input from the JSON and parse it to reuturn the correct tensor for inference
    :param input_json:
    :return: Tensor
    """
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transform to the input image and add a batch dimension
    img = Image.open(input_json.image)
    return transform(img).unsqueeze(0)


def post_processing(output_tensor):
    """
    Take the tensor output and return a JSON
    :param output_tensor: output of the model
    :return: JSON
    """

    return {"prediction": output_tensor.tolist()}
