import time
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
from logger import Logger
from torch.profiler import profile, ProfilerActivity, record_function


# -------------------------------------------
# Model-specific configuration
# -------------------------------------------
MODEL_NAME = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
id2label = model.config.id2label

# -------------------------------------------
# Logger setup
# -------------------------------------------
logger = Logger(model_name=MODEL_NAME)


router = APIRouter()


@router.post("/run")
async def run_model(file: UploadFile = File(...), ue_id: str = Form(...)):
    if ue_id is None:
        return JSONResponse(
            content={"error": "UE_ID is required."},
            status_code=400,
        )

    try:
        start_time = time.time()

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
            input_size=0,
            execution_duration=execution_duration,
        )

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "predictions": predictions,
                "input_size_bytes": 0,
                "execution_duration": execution_duration,
            }
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            content={"error": "Failed to process the image. {e}".format(e=str(e))},
            status_code=500,
        )


@router.post("/profile")
async def profile_service(file: UploadFile = File(...), ue_id: str = Form(...)):
    """
    Endpoint to profile the AI model execution.
    """
    if ue_id is None:
        return JSONResponse(
            content={"error": "UE_ID is required."},
            status_code=400,
        )

    profile_activities = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
        ProfilerActivity.MTIA,
        ProfilerActivity.XPU,
    ]

    try:
        profile_start_time = time.time()

        # Process the input image
        image = Image.open(file.file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with profile(
            activities=profile_activities,
            profile_memory=True,
            # record_shapes=True,
        ) as prof:
            with record_function("model_inference"):
                # Run the model
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

        execution_duration = time.time() - profile_start_time

        # Format profiler data into JSON
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # get only t
        profile_event = prof.key_averages()[0]

        profile_result = {
            "name": profile_event.key,
            "device_type": str(profile_event.device_type),
            "device_name": str(profile_event.use_device),
            "cpu_memory_usage": max([abs(e.cpu_memory_usage) for e in prof.key_averages()]),
            "self_cpu_memory_usage": max([abs(e.self_cpu_memory_usage) for e in prof.key_averages()]),
            "device_memory_usage": max([abs(e.device_memory_usage) for e in prof.key_averages()]),
            "self_device_memory_usage": max([abs(e.self_device_memory_usage) for e in prof.key_averages()]),
            "cpu_time_total": profile_event.cpu_time_total,
            "self_cpu_time_total": profile_event.self_cpu_time_total,
            "device_time_total": profile_event.device_time_total,
            "self_device_time_total": profile_event.self_device_time_total,
        }

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "node_id": logger.node_id,
                "k8s_pod_name": logger.k8s_pod_name,
                "profile_result": profile_result,
                "ai_result": predictions,
                "execution_duration": execution_duration,
            }
        )

    except Exception as e:
        print(f"Error processing request: {e}")
        return JSONResponse(
            content={"error": f"Failed to process the request. {e}"},
            status_code=500,
        )


# Below are the model input and output specifications to be used by the `/help` endpoint
MODEL_INPUT_FORM_SPEC = {
    "file": {
        "type": "file upload",
        "description": "The image file to be classified.",
        "required": True,
        "example": "puppy.png",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "input_size_bytes": "size of the input image in bytes",
    "execution_duration": "time taken for the model to process the image in seconds",
    "predictions": [
        {
            "category_id": "category id",
            "label": "category label",
            "probability": "probability value",
        }
    ],
}

PROFILE_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "profile_result": {
        "name": "name of the profile event",
        "device_type": "type of device used (e.g., CPU, GPU, ...)",
        "device_name": "name of the device used",
        "cpu_memory_usage": "CPU memory usage in bytes",
        "self_cpu_memory_usage": "self CPU memory usage in bytes",
        "device_memory_usage": "device memory usage in bytes",
        "self_device_memory_usage": "self device memory usage in bytes",
        "cpu_time_total": "total CPU time in microseconds",
        "self_cpu_time_total": "self total CPU time in microseconds",
        "device_time_total": "total device time in microseconds",
        "self_device_time_total": "self total device time in microseconds",
    },
    "ai_result": [
        {
            "category_id": "category id",
            "label": "category label",
            "probability": "probability value",
        }
    ],
    "execution_duration": "time taken for the model to process the image in seconds",
}
