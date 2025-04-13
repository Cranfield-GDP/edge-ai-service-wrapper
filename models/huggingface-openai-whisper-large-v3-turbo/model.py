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
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "openai/whisper-large-v3-turbo"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(file: UploadFile = File(...), ue_id: str = Form(...)):
    try:
        # Prepare the model input
        audio_bytes = await file.read()

        # Perform inference
        result = pipe(audio_bytes)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": result["text"],
            }
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            content={"error": "Failed to process the audio. {e}".format(e=str(e))},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(file: UploadFile = File(...), ue_id: str = Form(...)):
    """
    Endpoint to profile the AI model execution.
    """
    try:
        # Prepare the model input
        audio_bytes = await file.read()

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                result = pipe(audio_bytes)

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": result["text"],
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
        "description": "The audio file to be transcribed.",
        "required": True,
        "example": "sample.wav",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": "transcribed text from the audio",
}