from abc import ABC, abstractmethod
import torch
device = "cuda" if torch.cuda.is_available() else 'cpu'


class ZooModel(ABC):
    def __init__(self, input_json):
        self.description = "This is a sample model description"
        self.http_request = {"image": "./images/sample.jpg",
                             "checkpoint": "./checkpoints/checkpoint.pth",
                             "model_name": "samplemodel"
                             }

        self.input_json = input_json
        self.model = self.get_model().to(device)
        self.model.eval()

    @abstractmethod
    def get_model(self):
        """Abstract method to get the model"""
        Exception("Not implemented")
        pass

    @abstractmethod
    def pre_processing(self):
        """Abstract method for pre-processing the input data"""
        Exception("Not implemented")
        pass

    @abstractmethod
    def post_processing(self, input_tensor):
        """Abstract method for post-processing the output data"""
        Exception("Not implemented")
        pass

    def predict(self):
        x = self.pre_processing().to(device)
        y = self.model(x)
        y1 = self.post_processing(y)
        return y1
