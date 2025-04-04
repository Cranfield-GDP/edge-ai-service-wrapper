import json
import os
import subprocess

from dotenv import load_dotenv
from utils import (
    get_docker_image_build_name,
    get_image_repository_full_url,
    get_hf_model_directory,
    SERVICE_DATA_JSON_NAME
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def push_docker_image_main(huggingface_model_name: str) -> None:
    # check if the docker image exists
    ai_service_image_name = get_docker_image_build_name(huggingface_model_name)

    try:
        subprocess.run(
            ["docker", "image", "inspect", ai_service_image_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print(f"The docker image '{ai_service_image_name}' does not exist.")
        print(
            "Please build the docker image first using the build_docker_image.py script."
        )
        exit(1)

    # --------------------------------
    # Push the docker image to docker hub
    # ---------------------------------
    docker_image_repository_url = get_image_repository_full_url(huggingface_model_name)

    print(f"Pushing the docker image '{docker_image_repository_url}' to Docker Hub...")
    subprocess.run(
        ["docker", "tag", ai_service_image_name, docker_image_repository_url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["docker", "push", docker_image_repository_url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(
        f"The docker image '{docker_image_repository_url}' has been pushed to Docker Hub."
    )

    # --------------------------------
    # Update the service_data.json file
    # ---------------------------------
    hf_model_directory = get_hf_model_directory(huggingface_model_name)
    service_data_json_path = os.path.join(
        hf_model_directory, SERVICE_DATA_JSON_NAME
    )
    with open(service_data_json_path, "r") as f:
        service_data = json.load(f)
    # update the image repository url
    service_data["image_repository_url"] = docker_image_repository_url
    with open(service_data_json_path, "w") as f:
        json.dump(service_data, f, indent=4)
    print(
        f"The service_data.json file has been updated with the new image repository URL: {docker_image_repository_url}"
    )


if __name__ == "__main__":
    hf_model_name = input(
        "Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): "
    )
    ai_service_image_name = get_docker_image_build_name(hf_model_name)
    push_docker_image_main(ai_service_image_name)
