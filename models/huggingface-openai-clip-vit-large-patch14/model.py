# import server utils
from ai_server_utils import (
    get_image_classification_results_from_model_output_logits,
    profile_activities,
    prepare_profile_results,
)
# import profile utils
from torch.profiler import profile, record_function

# import necessary libs for AI model inference and request handling
import torch
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "openai/clip-vit-large-patch14"
# note that "use_fast" field will lead to a bug for the processor.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(file: UploadFile = File(...), text_prompts: list[str] = Form(...), ue_id: str = Form(...)):
    try:
        # Prepare the model input
        image = Image.open(file.file).convert("RGB")
        inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(device)

        # Perform inference
        with torch.no_grad():
            model_outputs = model(**inputs)

        # Process the model_outputs
        logits_per_image = model_outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()

        # Map probabilities to text prompts
        results = [
            {"text_prompt": text, "probability": float(prob)}
            for text, prob in zip(text_prompts, probs[0])
        ]

        # Sort results by probability (optional)
        results = sorted(results, key=lambda x: x["probability"], reverse=True)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": results,
            }
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(
            content={"error": "Failed to process the image. {e}".format(e=str(e))},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(file: UploadFile = File(...), text_prompts: list[str] = Form(...), ue_id: str = Form(...)):
    """
    Endpoint to profile the AI model execution.
    """
    try:
        # Prepare the model input
        image = Image.open(file.file).convert("RGB")
        inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(device)

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                with torch.no_grad():
                    model_outputs = model(**inputs)

        profile_result = prepare_profile_results(prof)

        # Process the model_outputs
        logits_per_image = model_outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()

        # Map probabilities to text prompts
        results = [
            {"text_prompt": text, "probability": float(prob)}
            for text, prob in zip(text_prompts, probs[0])
        ]

        # Sort results by probability (optional)
        results = sorted(results, key=lambda x: x["probability"], reverse=True)


        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": results,
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
    },
    "text_prompts": {
        "type": "list",
        "description": "A list of text prompts for the model to classify the image against.",
        "required": True,
        "example": ["a photo of a cat", "a photo of a dog"],
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