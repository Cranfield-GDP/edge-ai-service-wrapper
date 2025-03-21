import os
import time
import uuid
import requests
import psutil
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

# Get the IP and port of the edge logger server from environment variables or configuration
EDGE_LOGGER_SERVER_IP = os.getenv("EDGE_LOGGER_SERVER_IP", "localhost")
EDGE_LOGGER_SERVER_PORT = os.getenv("EDGE_LOGGER_SERVER_PORT", "8080")

app = FastAPI()

# Load the pre-trained ResNet-50 model and image processor
model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = ResNetForImageClassification.from_pretrained(model_name)
model.eval()

# Retrieve the mapping from category IDs to labels
id2label = model.config.id2label

@app.post("/run")
async def run_model(file: UploadFile = File(...)):
    start_time = time.time()
    input_bytes = await file.read()
    input_size = len(input_bytes)

    # Process the input image
    image = Image.open(file.file).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    output_size = probabilities.element_size() * probabilities.nelement()

    execution_time = time.time() - start_time

    # Gather resource usage
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = None  # Implement GPU usage retrieval if applicable

    # Prepare log data
    log_data = {
        "container_id": str(uuid.uuid4()),
        "model_name": model_name,
        "input_type": file.content_type,
        "input_size": input_size,
        "output_type": "tensor",
        "output_size": output_size,
        "execution_time": execution_time,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "gpu_usage": gpu_usage,
    }

    # Send log data to the logging server
    try:
        requests.post(f"http://{EDGE_LOGGER_SERVER_IP}:{EDGE_LOGGER_SERVER_PORT}/log", json=log_data)
    except requests.exceptions.RequestException as e:
        print(f"Logging failed: {e}")

    # Return the top 5 predictions with labels
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    predictions = []
    for i in range(top5_prob.size(0)):
        category_id = top5_catid[i].item()
        predictions.append({
            "category_id": category_id,
            "label": id2label[category_id],
            "probability": top5_prob[i].item()
        })

    return JSONResponse(content={"predictions": predictions, "execution_time": execution_time})

@app.get("/help")
def get_help():
    return {
        "endpoints": {
            "/run": {
                "method": "POST",
                "description": "Accepts an image file and returns model predictions.",
                "parameters": {
                    "file": "Image file to be processed."
                },
                "response": {
                    "predictions": "List of top 5 predictions with category IDs and probabilities."
                }
            },
            "/resource": {
                "method": "GET",
                "description": "Returns current CPU, RAM, and GPU usage of the container.",
                "response": {
                    "cpu_usage": "CPU usage percentage.",
                    "ram_usage": "RAM usage percentage.",
                    "gpu_usage": "GPU usage percentage (if applicable)."
                }
            }
        }
    }

@app.get("/resource")
def get_resource_usage():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = None  # Implement GPU usage retrieval if applicable

    return {
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "gpu_usage": gpu_usage
    }
