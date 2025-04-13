# import server utils
from ai_server_utils import (
    profile_activities,
    prepare_profile_results,
)

# import profile utils
from torch.profiler import profile, record_function

# import necessary libs for AI model inference and request handling
import torch
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from transformers import pipeline

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `classifier` are defined here.
# --------------------------------
MODEL_NAME = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(sequence: str = Form(...), candidate_labels: list[str] = Form(...), ue_id: str = Form(...)):
    try:
        # Perform inference
        result = classifier(sequence, candidate_labels)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": {
                    "labels": result['labels'],
                    "scores": result['scores'],
                    "sequence": result['sequence'],
                },
            }
        )
    except Exception as e:
        print(f"Error processing text: {e}")
        return JSONResponse(
            content={"error": "Failed to process the text. {e}".format(e=str(e))},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(sequence: str = Form(...), candidate_labels: list[str] = Form(...), ue_id: str = Form(...)):
    """
    Endpoint to profile the AI model execution.
    """
    try:
        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                result = classifier(sequence, candidate_labels)

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": {
                    "labels": result['labels'],
                    "scores": result['scores'],
                    "sequence": result['sequence'],
                },
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
    "sequence": {
        "type": "string",
        "description": "The sequence to be classified.",
        "required": True,
        "example": "one day I will see the world",
    },
    "candidate_labels": {
        "type": "list",
        "description": "The candidate labels for classification.",
        "required": True,
        "example": ["travel", "world", "day"],
    },
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": {
        "labels": "list of candidate labels",
        "scores": "list of scores corresponding to each label",
        "sequence": "the input sequence",
    },
}