# import server utils
from ai_server_utils import (
    profile_activities,
    prepare_profile_results,
)
# import profile utils
from torch.profiler import profile, record_function

# import necessary libs for AI model inference and request handling
import torch
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(file: UploadFile = File(...), ue_id: str = Form(...), text: str = Form(None)):
    try:
        # Prepare the model input
        image = Image.open(file.file).convert("RGB")
        if text:
            inputs = processor(image, text, return_tensors="pt").to(device)
        else:
            inputs = processor(image, return_tensors="pt").to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model.generate(**inputs)

        # Process the model outputs
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": caption,
            }
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            content={"error": "Failed to process the image. {e}".format(e=str(e))},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(file: UploadFile = File(...), ue_id: str = Form(...), text: str = Form(None)):
    """
    Endpoint to profile the AI model execution.
    """
    try:
        # Prepare the model input
        image = Image.open(file.file).convert("RGB")
        if text:
            inputs = processor(image, text, return_tensors="pt").to(device)
        else:
            inputs = processor(image, return_tensors="pt").to(device)

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                with torch.no_grad():
                    outputs = model.generate(**inputs)

        profile_result = prepare_profile_results(prof)

        # Process the model outputs
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": caption,
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
        "description": "The image file to be captioned.",
        "required": True,
        "example": "puppy.png",
    },
    "text": {
        "type": "string",
        "description": "Optional text for conditional image captioning.",
        "required": False,
        "example": "a photography of",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": "generated caption for the image",
}