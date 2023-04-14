# Docker Inference Template

This repository contains a template for creating a Docker image that can be used for deploying deep learning models for inference.

### Mantainers
- Rosario Di Carlo
- Roberto Morelli
### Requirements

- Docker installed on your local machine

### Usage

- Clone this repository to your local machine:
    ```
    git clone https://github.com/example/docker-inference-template.git
    ```

- Build the Docker image:
   
   ```
    docker build -t modelstore .
    ```

- Run the Docker container:
    ```
    docker run modelstore --json json/input.json
    ```

### Add custom models
- Add a new module in ./models <br>
  ex. transformer.py


- Add three methods
  - **get_model()**: A method that return your custom model
  - **pre_processing(input_file)**: A method that takes your model input and perform prepocessing
  - **post_processing(output_tensor)**: A method that takes your model output and save the results
  

- Put your model checkpoint in the ./checkpoints folder


- Define your input json-file and save it in the json folder<br>
    ```
   {
      "image": "./images/sample.jpg",
      "checkpoint": "./checkpoints/resnet18-5c106cde.pth",
      "model_name": "resnet"
    }
    ```
   NB: model_name should be the same of your module name in models folder. Ex. transformer.
