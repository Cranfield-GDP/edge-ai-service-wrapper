import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from logger import Logger

# -------------------------------------------
# Model setup
# -------------------------------------------
ai_model_endpoint_spec = {
    "model_input_form_spec": None,
    "model_output_json_spec": None,
    "profile_output_json_spec": None,
}


# -------------------------------------------
# App Lifespan setup
# -------------------------------------------
# Record the script start time (when uvicorn starts the process)
SCRIPT_START_TIME = time.time()
INITIALIZATION_DURATION = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):

    global INITIALIZATION_DURATION
    global SCRIPT_START_TIME
    global ai_model_endpoint_spec

    # Load the AI model
    print("Loading AI model...")
    from model import (
        MODEL_INPUT_FORM_SPEC,
        MODEL_OUTPUT_JSON_SPEC,
        PROFILE_OUTPUT_JSON_SPEC,
        router as model_router,
    )

    ai_model_endpoint_spec["model_input_form_spec"] = MODEL_INPUT_FORM_SPEC
    ai_model_endpoint_spec["model_output_json_spec"] = MODEL_OUTPUT_JSON_SPEC
    ai_model_endpoint_spec["profile_output_json_spec"] = PROFILE_OUTPUT_JSON_SPEC

    app.include_router(model_router, prefix="/model", tags=["AI Model"])

    # Record the initialization duration
    INITIALIZATION_DURATION = time.time() - SCRIPT_START_TIME

    print(f"AI model loaded in {INITIALIZATION_DURATION:.2f} seconds.")

    yield

    # Clean up the models and release the resources
    ai_model_endpoint_spec.clear()


# -------------------------------------------
# FastAPI application setup
# -------------------------------------------
app = FastAPI(lifespan=lifespan)


@app.get("/get_ue_log")
def get_log(ue_id: str):
    # Retrieve logs for the specified UE_ID
    logger = Logger.get_instance()
    if logger is None:
        return JSONResponse(
            content={"error": "Logger not initialized."},
            status_code=500,
        )

    log_data = logger.get_ue_run_log(ue_id=ue_id)
    if log_data is None:
        return JSONResponse(
            content={"error": f"No logs found for UE_ID: {ue_id}"},
            status_code=404,
        )
    return JSONResponse(content=log_data)


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
    global ai_model_endpoint_spec
    return {
        "endpoints": {
            "/model/run": {
                "method": "POST",
                "description": "Executes the AI model with the provided input data.",
                "parameters": {
                    "ue_id": "User Equipment ID (string) for tracking the request.",
                    **ai_model_endpoint_spec["model_input_form_spec"],
                },
                "response": {
                    "ue_id": "User Equipment ID (string) for tracking the request.",
                    "execution_duration": "Time taken to execute the model (in seconds).",
                    **ai_model_endpoint_spec["model_output_json_spec"],
                },
            },
            "/model/profile": {
                "method": "POST",
                "description": "Profiles the AI model execution.",
                "parameters": {
                    "ue_id": "User Equipment ID (string) for tracking the request.",
                    **ai_model_endpoint_spec["model_input_form_spec"],
                },
                "response": {
                    "ue_id": "User Equipment ID (string) for tracking the request.",
                    "profile_results": "Profiling results of the AI model execution.",
                    **ai_model_endpoint_spec["profile_output_json_spec"],
                },
            },
            "/get_ue_log": {
                "method": "GET",
                "description": "Retrieves logs for a specific UE_ID.",
                "parameters": {
                    "ue_id": "User Equipment ID (string) to retrieve logs for."
                },
                "response": {
                    "node_id": "Name of the computation node running the model.",
                    "k8s_pod_name": "Name of the Kubernetes pod running the model.",
                    "model_name": "Name of the AI model.",
                    "ue_id": "User Equipment ID (string) for which logs are retrieved.",
                    "total_input_size": "Total size of input data processed for the UE_ID (in bytes).",
                    "total_execution_duration": "Total time taken for all executions for the UE_ID (in seconds).",
                    "total_executions": "Total number of executions for the UE_ID.",
                    "average_execution_duration": "Average time taken for each execution for the UE_ID (in seconds).",
                    "latest_run": {
                        "input_size": "Size of the latest input data processed (in bytes).",
                        "execution_duration": "Time taken for the latest execution (in seconds).",
                        "timestamp": "Timestamp of the latest execution"
                        "(in seconds since epoch).",
                    },
                },
            },
            "/initialization_duration": {
                "method": "GET",
                "description": "Retrieves the initialization duration of the AI model.",
                "response": {
                    "initialization_duration": "Time taken to initialize the model (in seconds).",
                    "script_start_time": "Timestamp when the script started (in seconds since epoch).",
                },
            }
        }
    }
