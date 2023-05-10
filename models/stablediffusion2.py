import torch
from PIL import Image
from torchvision import models, transforms
from diffusers import StableDiffusionPipeline
from models.basemodel import ZooModel


class StableDiffusion2(ZooModel):
    """
    StableDiffusion2 model
    """

    def __init__(self, input_json):
        super().__init__(input_json)
        self.description = "StableDiffusion2 model allows to generate images from text"
        self.http_request = {
            "positive_prompt": "An image of an helicopter flying in front of a mountain, high quality",
            "negative_prompt": "low quality",
            "model_name": "stablediffusion2",
            "save_path": "./sd.png"
        }

    def load_model(self):
        """
        Load the model
        :return: model
        """
        model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                        torch_dtype=torch.float32)
        self.model = model.to(self.device)

    def pre_processing(self):
        """
        Take self.input_json and parse it to return the correct arguments for the model inference
        :return: args, kargs
        """

        args = {"positve_prompt": self.input_json.data["positive_prompt"]}
        kargs = {"negative_prompt": self.input_json.data["negative_prompt"]}
        return args, kargs

    def post_processing(self, output_tensor):
        """
        Take output_tensor and return a JSON
        :return: JSON
        """
        img = output_tensor.images[0]
        out_file = self.input_json.data["save_path"]
        img.save(out_file)
        return {"img_generated": out_file}
