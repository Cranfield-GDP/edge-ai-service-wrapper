# import model utilities
from ai_server_utils import process_model_output_logits, profile_model_run

# import necessary libs for AI model and request handling
import torch
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image


# Model-specific configuration
MODEL_NAME = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
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
        model_outputs, profile_result = profile_model_run(
            model_inputs=inputs,
            model=model,
        )

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
