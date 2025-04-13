# import server utils
from ai_server_utils import (
    profile_activities,
    prepare_profile_results,
)

# import profile utils
from torch.profiler import profile, record_function

# import necessary libs for AI model inference and request handling
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from flair.data import Sentence
from flair.models import SequenceTagger

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "flair/ner-english-fast"
tagger = SequenceTagger.load(MODEL_NAME)

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(text: str = Form(...), ue_id: str = Form(...)):
    try:
        # Prepare the sentence
        sentence = Sentence(text)

        # Perform inference
        tagger.predict(sentence)

        # Extract NER results
        ner_results = [str(entity) for entity in sentence.get_spans('ner')]
        
        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": ner_results,
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
        # Prepare the sentence
        sentence = Sentence(text)

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                tagger.predict(sentence)

        # Extract NER results
        ner_results = [str(entity) for entity in sentence.get_spans('ner')]

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": ner_results,
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
        "example": "George Washington went to Washington.",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": "Named entity recognition results",
}