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
from transformers import TessarTokenizer, BartForConditionalGeneration
import pandas as pd

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Model-specific configuration
# make sure the variables `MODEL_NAME` and `model` are defined here.
# --------------------------------
MODEL_NAME = "SVECTOR-CORPORATION/Tessar-largest"
tokenizer = TessarTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Initialize the FastAPI router
router = APIRouter()

@router.post("/run")
async def run_model(table_data: dict = Form(...), query: str = Form(...), ue_id: str = Form(...)):
    try:
        # Prepare table data
        table = pd.DataFrame.from_dict(table_data)

        # Encode the input
        encoding = tokenizer(table=table, query=query, return_tensors="pt").to(device)

        # Perform inference
        outputs = model.generate(**encoding)

        # Decode the results
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "model_results": result,
            }
        )
    except Exception as e:
        print(f"Error processing request: {e}")
        return JSONResponse(
            content={"error": f"Failed to process the request. {e}"},
            status_code=500,
        )

@router.post("/profile_run")
async def profile_run(table_data: dict = Form(...), query: str = Form(...), ue_id: str = Form(...)):
    """
    Endpoint to profile the AI model execution.
    """
    try:
        # Prepare table data
        table = pd.DataFrame.from_dict(table_data)

        # Encode the input
        encoding = tokenizer(table=table, query=query, return_tensors="pt").to(device)

        # perform profiling
        with profile(
            activities=profile_activities,
            profile_memory=True,
        ) as prof:
            with record_function("model_run"):
                outputs = model.generate(**encoding)

        # Decode the results
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        profile_result = prepare_profile_results(prof)

        return JSONResponse(
            content={
                "ue_id": ue_id,
                "profile_result": profile_result,
                "model_results": result,
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
    "table_data": {
        "type": "string",
        "description": "The table data in JSON format.",
        "required": True,
        "example": '{"year": [1896, 1900, 1904, 2004, 2008, 2012], "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]}',
    },
    "query": {
        "type": "string",
        "description": "The query to be answered based on the table data.",
        "required": True,
        "example": "In which year did beijing host the Olympic Games?",
    }
}

MODEL_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "model_results": "Table question answering results",
}