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
from kokoro import KPipeline
import soundfile as sf
import base64
import io

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `pipeline` are defined here.
# --------------------------------
MODEL_NAME = "hexgrad/Kokoro-82M"


# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(lang_code: str = Form(...), voice: str = Form(...), speed: float = Form(...), text: str = Form(...), ue_id: str = Form(...)):
    try:

        pipeline = KPipeline(lang_code=lang_code, device=device)

        # Prepare the model input
        generator = pipeline(text, voice=voice, speed=1.0)

        # Create an in-memory buffer for the WAV file
        wav_buf = io.BytesIO()

        # Open the buffer in write mode with soundfile
        with sf.SoundFile(wav_buf, mode="w", samplerate=24000, channels=1, format="WAV") as wav_file:
            for _, _, audio in generator:
                # Write audio data to the buffer
                wav_file.write(audio)

        # Ensure the buffer is ready for reading
        wav_buf.seek(0)

        # Encode the audio data to base64
        audio_data = base64.b64encode(wav_buf.read()).decode("utf-8")

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": audio_data,
            }
        )
    except Exception as e:
        print(f"Error processing text: {e}")
        return JSONResponse(
            content={"error": "Failed to process the text. {e}".format(e=str(e))},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(lang_code: str = Form(...), voice: str = Form(...), speed: float = Form(...), text: str = Form(...), ue_id: str = Form(...)):
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

                pipeline = KPipeline(lang_code=lang_code, device=device)

                # Prepare the model input
                generator = pipeline(text, voice=voice, speed=1.0)

                # Create an in-memory buffer for the WAV file
                wav_buf = io.BytesIO()

                # Open the buffer in write mode with soundfile
                with sf.SoundFile(wav_buf, mode="w", samplerate=24000, channels=1, format="WAV") as wav_file:
                    for _, _, audio in generator:
                        # Write audio data to the buffer
                        wav_file.write(audio)
                
                # Ensure the buffer is ready for reading
                wav_buf.seek(0)

                # Encode the audio data to base64
                audio_data = base64.b64encode(wav_buf.read()).decode("utf-8")

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": audio_data,
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
        "description": "The text input to be converted to speech.",
        "required": True,
        "example": "Hello, world!",
    },
    "lang_code": {
        "type": "string",
        "description": "The language code for the text input.",
        "required": True,
        "example": "en",
    },
    "voice": {
        "type": "string",
        "description": "The voice to be used for speech synthesis.",
        "required": True,
        "example": "en-US-Wavenet-D",
    },
    "speed": {
        "type": "float",
        "description": "The speed of the speech synthesis. Normal speed is 1.0.",
        "required": True,
        "example": 1.0,
    },
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": [
        {
            "audio_data": "base64 encoded audio data",
        }
    ],
}