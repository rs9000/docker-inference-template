from abc import ABC, abstractmethod
import torch


class ZooModel(ABC):
    def __init__(self, input_json):
        self.description = "This is a sample model description"
        self.http_request = {"image": "./images/sample.jpg",
                             "checkpoint": "./checkpoints/checkpoint.pth",
                             "model_name": "samplemodel"
                             }

        self.input_json = input_json
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model().to(self.device)

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
        args, kargs = self.pre_processing()
        y = self.model(*args.values(), **kargs)
        y1 = self.post_processing(y)
        return y1
