from PIL import Image
from torchvision import models, transforms
from models.basemodel import ZooModel


class EfficientNetB0(ZooModel):
    """
    EfficientNetB0 model
    """

    def __init__(self, input_json):
        super().__init__(input_json)
        self.description = "EfficientNetB0 model trained on ImageNet"
        self.http_request = {"image": "./images/sample.jpg",
                             "checkpoint": "./checkpoints/efficientnet_b0_rwightman-3dd342df.pth",
                             "model_name": "efficientnet_b0"
                             }

    def get_model(self):
        """
        Load the model
        :return: model
        """
        model = models.efficientnet_b0(pretrained=False, num_classes=1000)
        return model

    def pre_processing(self):
        """
        Take self.input_json and parse it to return the correct tensor for inference
        :return: Tensor
        """
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Apply the transform to the input image and add a batch dimension
        img = Image.open(self.input_json.image)
        return transform(img).unsqueeze(0)

    def post_processing(self, output_tensor):
        """
        Take output_tensor and return a JSON
        :return: JSON
        """

        return {"prediction": output_tensor.tolist()}
