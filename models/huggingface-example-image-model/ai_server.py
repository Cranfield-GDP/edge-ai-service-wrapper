import json
import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from ai_server_utils import (
    PROFILE_OUTPUT_JSON_SPEC,
    NODE_ID,
    K8S_POD_NAME,
)


# -------------------------------------------
# App Lifespan setup
# -------------------------------------------
# Record the script start time (when uvicorn starts the process)
SCRIPT_START_TIME = time.time()
INITIALIZATION_DURATION = 0.0
service_endpoint_specs = {
    "model_input_form_spec": None,
    "model_output_json_spec": None,
    "profile_output_json_spec": None,
    "xai_model_input_form_spec": None,
    "xai_model_output_json_spec": None,
    "xai_profile_output_json_spec": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):

    global INITIALIZATION_DURATION
    global SCRIPT_START_TIME
    global service_endpoint_specs

    # Load the AI model
    print("Loading AI model...")
    from model import (
        MODEL_INPUT_FORM_SPEC,
        MODEL_OUTPUT_JSON_SPEC,
        router as model_router,
    )

    service_endpoint_specs["model_input_form_spec"] = MODEL_INPUT_FORM_SPEC
    service_endpoint_specs["model_output_json_spec"] = MODEL_OUTPUT_JSON_SPEC
    service_endpoint_specs["profile_output_json_spec"] = PROFILE_OUTPUT_JSON_SPEC

    app.include_router(model_router, prefix="/model", tags=["AI Model"])

    # Load the XAI model
    if os.path.exists(os.path.dirname(__file__) + "/xai_model.py"):
        print("Loading XAI model...")
        from xai_model import (
            XAI_OUTPUT_JSON_SPEC,
            router as xai_model_router,
        )

        # by default, the xai_model input form spec is the same as the model input form spec
        service_endpoint_specs["xai_model_input_form_spec"] = MODEL_INPUT_FORM_SPEC
        service_endpoint_specs["xai_model_output_json_spec"] = MODEL_OUTPUT_JSON_SPEC
        service_endpoint_specs["xai_model_output_json_spec"].update(
            XAI_OUTPUT_JSON_SPEC
        )
        service_endpoint_specs["xai_profile_output_json_spec"] = (
            PROFILE_OUTPUT_JSON_SPEC
        )
        service_endpoint_specs["xai_profile_output_json_spec"].update(
            XAI_OUTPUT_JSON_SPEC
        )

        app.include_router(xai_model_router, prefix="/xai_model", tags=["XAI Model"])

    # Record the initialization duration
    INITIALIZATION_DURATION = time.time() - SCRIPT_START_TIME

    print(f"AI service loaded in {INITIALIZATION_DURATION:.2f} seconds.")

    yield

    # Clean up the models and release the resources
    service_endpoint_specs.clear()


# -------------------------------------------
# FastAPI application setup
# -------------------------------------------
app = FastAPI(lifespan=lifespan)


# -------------------------------------------
# Middlewares
# -------------------------------------------
@app.middleware("http")
async def prepare_header_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-NODE-ID"] = NODE_ID
    response.headers["X-K8S-POD-NAME"] = K8S_POD_NAME
    return response


# -------------------------------------------
# General Endpoints
# -------------------------------------------
@app.get("/initialization_duration")
def get_initialization_duration():
    """
    Endpoint to retrieve the initialization duration of the AI model.
    """
    global INITIALIZATION_DURATION

    if INITIALIZATION_DURATION == 0.0:
        return JSONResponse(
            content={"error": "Model not initialized."},
            status_code=500,
        )
    return JSONResponse(
        content={
            "initialization_duration": INITIALIZATION_DURATION,
            "script_start_time": SCRIPT_START_TIME,
        }
    )


@app.get("/help")
def get_help():
    global service_endpoint_specs
    help_info = {
        "endpoints": {
            "/model/run": {
                "method": "POST",
                "description": "Executes the AI model with the provided input data.",
                "parameters": {
                    "ue_id": "User Equipment ID (string) for tracking the request.",
                    **service_endpoint_specs["model_input_form_spec"],
                },
                "response": {
                    "ue_id": "User Equipment ID (string) for tracking the request.",
                    **service_endpoint_specs["model_output_json_spec"],
                },
            },
            "/model/profile_run": {
                "method": "POST",
                "description": "Profiles the AI model execution.",
                "parameters": {
                    "ue_id": "User Equipment ID (string) for tracking the request.",
                    **service_endpoint_specs["model_input_form_spec"],
                },
                "response": {
                    "ue_id": "User Equipment ID (string) for tracking the request.",
                    "profile_result": "Profiling results of the AI model execution.",
                    **service_endpoint_specs["profile_output_json_spec"],
                },
            },
            "/initialization_duration": {
                "method": "GET",
                "description": "Retrieves the initialization duration of the AI model.",
                "response": {
                    "initialization_duration": "Time taken to initialize the model (in seconds).",
                    "script_start_time": "Timestamp when the script started (in seconds since epoch).",
                },
            },
        }
    }

    if service_endpoint_specs["xai_model_input_form_spec"] is not None:
        help_info["endpoints"]["/xai_model/run"] = {
            "method": "POST",
            "description": "Executes the XAI model with the provided input data.",
            "parameters": {
                "ue_id": "User Equipment ID (string) for tracking the request.",
                **service_endpoint_specs["xai_model_input_form_spec"],
            },
            "response": {
                "ue_id": "User Equipment ID (string) for tracking the request.",
                **service_endpoint_specs["xai_model_output_json_spec"],
            },
        }

        help_info["endpoints"]["/xai_model/profile_run"] = {
            "method": "POST",
            "description": "Profiles the XAI model execution.",
            "parameters": {
                "ue_id": "User Equipment ID (string) for tracking the request.",
                **service_endpoint_specs["xai_model_input_form_spec"],
            },
            "response": {
                "ue_id": "User Equipment ID (string) for tracking the request.",
                "profile_result": "Profiling results of the XAI model execution.",
                **service_endpoint_specs["xai_profile_output_json_spec"],
            },
        }
    
    return JSONResponse(content=help_info)
