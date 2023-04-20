# Docker Inference modelstore

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


- Custom models must have the following 3 methods:
  - **get_model()**: A method that return your custom model
  - **pre_processing(input_file)**: A method that takes the JSON request and perform prepocessing to return a input for your model
  - **post_processing(output_tensor)**: A method that takes your model output and return a JSON output
  

- Put your model checkpoint in the ./checkpoints folder


- Rebuild the docker image


### Example of HTTP request to the inference server

```
POST http://127.0.0.1:8000/predict
{
  "image": "./images/sample.jpg",
  "checkpoint": "./checkpoints/resnet18-5c106cde.pth",
  "model_name": "resnet"
}
```

```
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
