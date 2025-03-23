import os
import subprocess

from dotenv import load_dotenv
from utils import (get_docker_image_build_name, get_image_repository_full_url)

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
        print("Please build the docker image first using the build_docker_image.py script.")
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
    print(f"The docker image '{docker_image_repository_url}' has been pushed to Docker Hub.")


if __name__ == "__main__":
    hf_model_name = input("Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): ")
    ai_service_image_name = get_docker_image_build_name(hf_model_name)
    push_docker_image_main(ai_service_image_name)