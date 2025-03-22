import os
import subprocess

from dotenv import load_dotenv
from utils import (get_docker_image_build_name)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

hf_model_name = input("Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): ")
ai_service_image_name = get_docker_image_build_name(hf_model_name)

# check if the docker image exists
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
docker_username = os.getenv("DOCKER_USERNAME")
docker_registry = os.getenv("DOCKER_REGISTRY", "docker.io")
docker_image_name = f"{docker_username}/{ai_service_image_name}"
docker_image_full_name = f"{docker_registry}/{docker_image_name}"
docker_image_full_name = docker_image_full_name.replace(" ", "_")

print(f"Pushing the docker image '{docker_image_full_name}' to Docker Hub...")
subprocess.run(
    ["docker", "tag", ai_service_image_name, docker_image_full_name],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
subprocess.run(
    ["docker", "push", docker_image_full_name],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
print(f"The docker image '{docker_image_full_name}' has been pushed to Docker Hub.")
