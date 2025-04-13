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
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Initialize the FastAPI router
router = APIRouter()

# Initialize the NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

@router.post("/run")
async def run_model(text: str = Form(...), ue_id: str = Form(...)):
    try:
        # Perform inference
        ner_results = nlp(text)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": str(ner_results),
            }
        )
    except Exception as e:
        print(f"Error processing text: {e}")
        return JSONResponse(
            content={"error": "Failed to process the text. {e}".format(e=str(e))},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(text: str = Form(...), ue_id: str = Form(...)):
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
                ner_results = nlp(text)

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": str(ner_results),
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
    "text": {
        "type": "string",
        "description": "The text to be analyzed for named entities.",
        "required": True,
        "example": "My name is Wolfgang and I live in Berlin.",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": "Named entity recognition results",
}