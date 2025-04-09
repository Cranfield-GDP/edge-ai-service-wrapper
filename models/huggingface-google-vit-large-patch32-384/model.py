# import server utils
from ai_server_utils import (
    process_model_output_logits,
    profile_activities,
    prepare_profile_results,
)
# import profile utils
from torch.profiler import profile, record_function

# import necessary libs for AI model inference and request handling
import torch
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "google/vit-large-patch32-384"
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME)
model.eval()


# Initialize the FastAPI router
router = APIRouter()


@router.post("/run")
async def run_model(file: UploadFile = File(...), ue_id: str = Form(...)):
    try:
        # Prepare the model input
        image = Image.open(file.file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Process the model outputs
        predictions = process_model_output_logits(model, outputs.logits)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": predictions,
            }
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            content={"error": "Failed to process the image. {e}".format(e=str(e))},
            status_code=500,
        )


@router.post("/profile_run")
async def profile_run(file: UploadFile = File(...), ue_id: str = Form(...)):
    """
    Endpoint to profile the AI model execution.
    """
    try:
        # Prepare the model input
        image = Image.open(file.file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                with torch.no_grad():
                    model_outputs = model(**inputs)

        profile_result = prepare_profile_results(prof)

        # Process the model outputs
        predictions = process_model_output_logits(model, model_outputs.logits)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": predictions,
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
    "model_results": [
        {
            "category_id": "category id",
            "label": "category label",
            "probability": "probability value",
        }
    ],
}