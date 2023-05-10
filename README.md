# Docker Inference ModelZoo
<img src="./images/logo.png" width="200">

This repository contains a Docker image template that can be used for deploying deep learning models for inference.

### Mantainers
- Rosario Di Carlo
- Roberto Morelli
### Requirements

- Docker installed on your local machine

### Usage

- Clone this repository to your local machine:
    ```
    git clone https://github.com/rs9000/docker-inference-template.git
    ```

- Build the Docker image:
   
   ```
    docker build -t modelstore .
    ```

- Run the Docker container:
    ```
    docker run -p 8000:8000 modelstore
    ```
    OR with local folders binding [optional] :
    ```
    docker run -p 8000:8000 -v checkpoints:/app/checkpoints -v images:/app/images modelstore
    ```


### Add custom models
- Add a new module in ./models <br>
  ex. transformer.py


- Custom models must be classes inheriting from ZooModel 
  and must define the following methods:

```python
from models.basemodel import ZooModel

class CustomModel(ZooModel):
    def __init__(self, input_json):
        super().__init__(input_json)
        self.description = "Custom model description"
        self.http_request = {"image": "./images/sample.jpg",
                             "checkpoint": "./checkpoints/checkpoint.pth",
                             "model_name": "custom_model"
                             }

    def get_model(self):
        """
        Load the model
        :return: model
        """
        model = # Custom model definition
        return model

    def pre_processing(self):
        """
        Take self.input_json and parse it to return the correct tensor for inference
        :return: Tensor
        """
        
        input_tensor = something(self.input_json)
        return input_tensor

    def post_processing(self, output_tensor):
        """
        Take output_tensor and return a JSON
        :return: JSON
        """
      
        output_json = something(output_tensor)
        return output_json
```

- Put your model checkpoint in the ./checkpoints folder
- Register your model in model_registry.py
- Rebuild the docker image


### Example of HTTP request to the inference server

```json
POST http://127.0.0.1:8000/predict
{
  "image": "./images/sample.jpg",
  "checkpoint": "./checkpoints/resnet18-5c106cde.pth",
  "model_name": "resnet"
}
```

```json
RESPONSE 200 OK
{
    "prediction": [
        [
            1.3546998500823975,
            1.220292329788208,
            ...
            1.6268298625946045,
        ]
    ]
}

```
