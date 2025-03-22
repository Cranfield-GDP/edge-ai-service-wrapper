import os
import time
import socket
from fastapi import Form
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

from logger import Logger

# -------------------------------------------
# Configuration
# -------------------------------------------
CONTAINER_ID = os.getenv("HOSTNAME", socket.gethostname())
CONTAINER_NAME = os.getenv("K8S_POD_NAME", CONTAINER_ID)
print(f"Container ID: {CONTAINER_ID}, Container Name: {CONTAINER_NAME}")

# -------------------------------------------
# Model loading
# -------------------------------------------
MODEL_NAME = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
id2label = model.config.id2label


# -------------------------------------------
# Logger setup
# -------------------------------------------
logger = Logger(
    container_id=CONTAINER_ID,
    container_name=CONTAINER_NAME,
    model_name=MODEL_NAME,
)


# -------------------------------------------
# FastAPI application setup
# -------------------------------------------
app = FastAPI()


@app.post("/run")
async def run_model(file: UploadFile = File(...), ue_id: str = Form(...)):
    if ue_id is None:
        return JSONResponse(
            content={"error": "UE_ID is required."},
            status_code=400,
        )

    try:
        start_time = time.time()
        input_bytes = await file.read()
        input_size = len(input_bytes)

        # Process the input image
        image = Image.open(file.file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

        # Return the top 5 predictions with labels
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        predictions = []
        for i in range(top5_prob.size(0)):
            category_id = top5_catid[i].item()
            predictions.append(
                {
                    "category_id": category_id,
                    "label": id2label[category_id],
                    "probability": top5_prob[i].item(),
                }
            )

        execution_duration = time.time() - start_time

        logger.add_ue_run_log(
            ue_id=ue_id,
            input_size=input_size,
            execution_duration=execution_duration,
        )

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "predictions": predictions,
                "execution_duration": execution_duration,
            }
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            content={"error": "Failed to process the image. {e}".format(e=str(e))},
            status_code=500,
        )


@app.get("/get_ue_log")
def get_log(ue_id: str):
    # Retrieve logs for the specified UE_ID
    log_data = logger.get_ue_run_log(ue_id=ue_id)
    if log_data is None:
        return JSONResponse(
            content={"error": f"No logs found for UE_ID: {ue_id}"},
            status_code=404,
        )
    return JSONResponse(content=log_data)


@app.get("/help")
def get_help():
    return {
        "endpoints": {
            "/run": {
                "method": "POST",
                "description": "Accepts an image file and a UE_ID, processes the image using the AI model, and returns the top 5 predictions.",
                "parameters": {
                    "file": "Image file to be processed (JPEG or PNG).",
                    "ue_id": "User Equipment ID (string) to identify the user.",
                },
                "response": {
                    "predictions": "List of top 5 predictions with category IDs, labels, and probabilities.",
                    "execution_duration": "Time taken to process the image and generate predictions (in seconds).",
                },
            },
            "/get_ue_log": {
                "method": "GET",
                "description": "Retrieves logs for a specific UE_ID.",
                "parameters": {
                    "ue_id": "User Equipment ID (string) to retrieve logs for."
                },
                "response": {
                    "container_id": "ID of the container running the model.",
                    "container_name": "Name of the container running the model.",
                    "model_name": "Name of the AI model.",
                    "ue_id": "User Equipment ID (string) for which logs are retrieved.",
                    "total_input_size": "Total size of input data processed for the UE_ID (in bytes).",
                    "total_execution_duration": "Total time taken for all executions for the UE_ID (in seconds).",
                    "total_executions": "Total number of executions for the UE_ID.",
                    "average_execution_duration": "Average time taken for each execution for the UE_ID (in seconds).",
                    "latest_run": {
                        "input_size": "Size of the latest input data processed (in bytes).",
                        "execution_duration": "Time taken for the latest execution (in seconds).",
                        "timestamp": "Timestamp of the latest execution"
                        "(in seconds since epoch).",
                    },
                },
            },
        }
    }
