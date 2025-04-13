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
from diffusers import StableDiffusionXLImg2ImgPipeline
from io import BytesIO
from PIL import Image

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `pipe` are defined here.
# --------------------------------
MODEL_NAME = "stabilityai/stable-diffusion-xl-refiner-1.0"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to(device)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(file: UploadFile = File(...), prompt: str = Form(...), ue_id: str = Form(...)):
    try:
        # Prepare the model input
        init_image = Image.open(file.file).convert("RGB")

        # Perform inference
        with torch.no_grad():
            image = pipe(prompt, image=init_image).images[0]

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": encode_image(image),
            }
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            content={"error": "Failed to process the image. {e}".format(e=str(e))},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(file: UploadFile = File(...), prompt: str = Form(...), ue_id: str = Form(...)):
    """
    Endpoint to profile the AI model execution.
    """
    try:
        # Prepare the model input
        init_image = Image.open(file.file).convert("RGB")

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                with torch.no_grad():
                    image = pipe(prompt, image=init_image).images[0]

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": encode_image(image),
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
        "description": "The initial image file for image-to-image generation.",
        "required": True,
        "example": "astronaut.png",
    },
    "prompt": {
        "type": "string",
        "description": "The text prompt to guide the image generation.",
        "required": True,
        "example": "a photo of an astronaut riding a horse on mars",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": "binary content of the generated image",
}