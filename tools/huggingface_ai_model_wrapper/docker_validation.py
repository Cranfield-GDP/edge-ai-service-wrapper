import os
import subprocess
from utils import (
    TARGET_FILES_TO_GENERATE,
    get_available_port,
    get_docker_container_run_name,
    get_docker_image_build_name,
    get_hf_model_directory,
    validate_hf_model_name,
)

def build_and_start_docker_container(huggingface_model_name: str) -> None:
    # --------------------------------
    # get the model directory
    # --------------------------------
    hf_model_directory = get_hf_model_directory(huggingface_model_name)
    # reset the model directory if it exists
    assert os.path.exists(
        hf_model_directory
    ), f"The model directory '{hf_model_directory}' does not exist."
    # check if all the required files exist
    for file_name in TARGET_FILES_TO_GENERATE:
        file_path = os.path.join(hf_model_directory, file_name)
        assert os.path.exists(
            file_path
        ), f"The required model file '{file_path}' does not exist."
    print("The model directory is valid.")

    # --------------------------------
    # Build the docker image
    # ---------------------------------
    subprocess.run(
        ["docker", "build", "-t", get_docker_image_build_name(huggingface_model_name), "."],
        cwd=hf_model_directory,
        check=True,
    )
    print(f"Docker image {get_docker_image_build_name(huggingface_model_name)} built successfully.")

    # --------------------------------
    # Run the docker container
    # ---------------------------------
    available_port = get_available_port()
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            get_docker_container_run_name(huggingface_model_name),
            "-p",
            f"{available_port}:8000",
            "--health-cmd", "python healthcheck.py",
            "--health-interval=5s",
            "--health-timeout=2s",
            "--health-retries=3",
            get_docker_image_build_name(huggingface_model_name),
        ],
        check=True,
    )
    print(f"Docker container {get_docker_container_run_name(huggingface_model_name)}-server started successfully.")
    print(f"Access the server at http://localhost:{available_port}/run")

if __name__ == "__main__":
    # --------------------------------
    # Validate the hugging face model
    # --------------------------------
    huggingface_model_name = input(
        "Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): "
    )
    # validate the hugging face model name
    assert validate_hf_model_name(
        huggingface_model_name
    ), f"The model '{huggingface_model_name}' is invalid."
    print(f"The model {huggingface_model_name} is valid.")

    build_and_start_docker_container(huggingface_model_name)
