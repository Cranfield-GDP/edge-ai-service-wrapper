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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(text: str = Form(...), ue_id: str = Form(...)):
    try:
        # Prepare the model input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        # Perform inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Process the model outputs
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        predicted_label = sentiment_map[predicted_class_id]

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": {
                    "predicted_label": predicted_label,
                    "logits": logits.tolist(),
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
async def profile_run(text: str = Form(...), ue_id: str = Form(...)):
    """
    Endpoint to profile the AI model execution.
    """
    try:
        # Prepare the model input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                with torch.no_grad():
                    logits = model(**inputs).logits

        profile_result = prepare_profile_results(prof)

        # Process the model outputs
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        predicted_label = sentiment_map[predicted_class_id]

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": {
                    "predicted_label": predicted_label,
                    "logits": logits.tolist(),
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
    "text": {
        "type": "string",
        "description": "The text to be classified.",
        "required": True,
        "example": "I love this product! It's amazing and works perfectly.",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": {
        "predicted_label": "predicted class label",
        "logits": "logits values",
    },
}