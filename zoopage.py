import json
import streamlit as st
import importlib
import requests as req
from model_registry import model_registry


def get_logs(logs, model_name, key):
    for model in logs["models"]:
        if model["model_name"] == model_name:
            return model[key]

    logs["models"].append({"model_name": model_name, key: 0})
    return 0


def set_logs(logs, model_name, key, value):
    for model in logs["models"]:
        if model["model_name"] == model_name:
            model[key] = value
    return logs


# Load the Models in the zoo
module_dir = 'models'
api_entrypoint = "http://localhost:8000/predict"
models = {}

for file_name, class_name in model_registry.items():
    module = importlib.import_module("models." + file_name)
    model_class = getattr(module, class_name)
    models[file_name] = model_class("")

st.title('Model Zoo')
st.image("./images/logo.png", width=250)
st.write(f"Found **{len(model_registry)}** models in the zoo")

search_term = st.text_input("", placeholder="Filter models by name")

# Filter the dictionary based on the search term
filtered_data = {
    key: value
    for key, value in model_registry.items()
    if search_term.lower() in key.lower()
}

with open("logs.json", "r") as f:
    logs = json.load(f)

for file_name, class_name in filtered_data.items():
    with st.expander(class_name):
        st.write(f"**Model**: {class_name}")
        st.write(f"**Description**: {models[file_name].description}")
        st.write(f"**Example request**:")
        header = {"POST": "/predict HTTP/1.1", "Host": "localhost:8000", "Content-Type": "application/json"}
        block = ""
        for k, v in header.items():
            block += f"{k}:  {v} \n"
        for k, v in models[file_name].http_request.items():
            block += f"{k}: {v} \n"
        st.code(block, language="http")

        if st.button("Execute HTTP Request", key=file_name):
            # HTTP request is triggered when the button is clicked
            with st.spinner("Loading..."):
                # HTTP request is triggered when the button is clicked
                response = req.post(api_entrypoint, json=models[file_name].http_request)

            # Process the response
            if response.status_code == 200:
                st.success("HTTP request successful!")
                st.code(response.text, language="json")
            else:
                st.error("HTTP request failed!")
                st.code(response.text)

            downloads_value = get_logs(logs, file_name, 'downloads')
            set_logs(logs, file_name, 'downloads', downloads_value + 1)

        downloads_value = get_logs(logs, file_name, 'downloads')
        st.write(f"Used **{downloads_value}** times by other users.")
        with open("logs.json", "w") as f:
            json.dump(logs, f)

st.divider()
st.caption("Powered by Leonardo Labs")
