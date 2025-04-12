# import server utils
from ai_server_utils import (
    encode_image,
    profile_activities,
    prepare_profile_results,
)

# import profile utils
from torch.profiler import profile, record_function

# import necessary libs for AI model inference and request handling
import torch
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "Ultralytics/YOLOv8"
model = YOLO("yolov8n-seg.pt")
model.eval()

# Initialize the FastAPI router
router = APIRouter()


def process_yolov8_segmentation_model_results(results):
    """
    Process the YOLOv8 segmentation model results.
    """
    rendered_image = results[0].plot(conf=True, pil=True, show=False, save=False)

    return results[0].summary(), rendered_image


@router.post("/run")
async def run_model(file: UploadFile = File(...), ue_id: str = Form(...)):
    try:
        # Prepare the model input
        image = Image.open(file.file)

        # Perform inference
        with torch.no_grad():
            results = model(image, device=device)

        model_results, visualization = process_yolov8_segmentation_model_results(
            results
        )

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": model_results,
                "visualization": encode_image(visualization),
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
        image = Image.open(file.file)

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                with torch.no_grad():
                    results = model(image, device=device)

        model_results, visualization = process_yolov8_segmentation_model_results(
            results
        )

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": model_results,
                "visualization": encode_image(visualization),
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
        "description": "The image file to be segmented.",
        "required": True,
        "example": "puppy.png",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": "The results of the segmentation model.",
    "visualization": {
        "type": "image",
        "description": "The segmented image with bounding boxes.",
    },
}
