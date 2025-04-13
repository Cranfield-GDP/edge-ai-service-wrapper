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
from sentence_transformers import SentenceTransformer

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME).to(device)
model.eval()

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(sentences: list[str] = Form(...), ue_id: str = Form(...)):
    try:

        # Perform inference
        with torch.no_grad():
            embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": embeddings.tolist(),
            }
        )
    except Exception as e:
        print(f"Error processing sentences: {e}")
        return JSONResponse(
            content={"error": "Failed to process the sentences. {e}".format(e=str(e))},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(sentences: list[str] = Form(...), ue_id: str = Form(...)):
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
                with torch.no_grad():
                    embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": embeddings.tolist(),
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
    "sntences": {
        "type": "list",
        "description": "List of sentences to be encoded.",
        "example": [
            "This is a sentence.",
            "This is another sentence."
        ]
    },
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": [
        "embedding vector for each sentence"
    ],
}