import json
import os
import subprocess
from utils import (
    NECESSARY_SERVICE_FILE_LIST,
    SERVICE_DATA_JSON_NAME,
    get_available_port,
    get_docker_container_run_name,
    get_docker_image_build_name,
    get_hf_model_directory,
    get_node_hostname,
    update_service_disk_size,
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
    for file_name in NECESSARY_SERVICE_FILE_LIST:
        file_path = os.path.join(hf_model_directory, file_name)
        assert os.path.exists(
            file_path
        ), f"The required model file '{file_path}' does not exist."
    print("The model directory is valid.")

    # --------------------------------
    # Build the docker image
    # ---------------------------------
    docker_image_build_name = get_docker_image_build_name(huggingface_model_name)
    subprocess.run(
        ["docker", "build", "-t", docker_image_build_name, "."],
        cwd=hf_model_directory,
        check=True,
    )
    print(f"Docker image {docker_image_build_name} built successfully.")

    # ----------------------------------
    # Save the disk size of the built docker image
    # ----------------------------------
    docker_image_size_bytes = subprocess.run(
        ["docker", "image", "inspect", docker_image_build_name, "--format={{.Size}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    update_service_disk_size(
        huggingface_model_name,
        int(docker_image_size_bytes),
    )

    # --------------------------------
    # Run the docker container
    # ---------------------------------
    available_port = get_available_port()
    profile_node_id = get_node_hostname()
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            get_docker_container_run_name(huggingface_model_name),
            "--env",
            f"NODE_ID={profile_node_id}",
            "-p",
            f"{available_port}:8000",
            "--health-cmd",
            "python healthcheck.py",
            "--health-interval=5s",
            "--health-timeout=2s",
            "--health-retries=3",
            get_docker_image_build_name(huggingface_model_name),
        ],
        check=True,
    )
    print(
        f"Docker container {get_docker_container_run_name(huggingface_model_name)}-server started successfully."
    )
    print(f"Access the server at http://localhost:{available_port}/run")


def update_container_memory_usage(huggingface_model_name: str) -> None:
    # --------------------------------
    # get the model directory
    # --------------------------------
    hf_model_directory = get_hf_model_directory(huggingface_model_name)
    # reset the model directory if it exists
    assert os.path.exists(
        hf_model_directory
    ), f"The model directory '{hf_model_directory}' does not exist."
    # check if all the required files exist
    for file_name in NECESSARY_SERVICE_FILE_LIST:
        file_path = os.path.join(hf_model_directory, file_name)
        assert os.path.exists(
            file_path
        ), f"The required model file '{file_path}' does not exist."
    print("The model directory is valid.")

    # --------------------------------
    # Read the service_data.json
    # ---------------------------------
    service_data_json_path = os.path.join(
        hf_model_directory, SERVICE_DATA_JSON_NAME
    )
    assert os.path.exists(
        service_data_json_path
    ), f"The service data json file '{service_data_json_path}' does not exist."

    with open(service_data_json_path, "r") as f:
        service_data = json.load(f)
    
    assert "profiles" in service_data and len(service_data["profiles"]) > 0, "No profiles found in service_data.json"
    
    node_id_list = [
        profile["node_id"] for profile in service_data["profiles"]
        if "node_id" in profile
    ]

    node_id_to_update = input("Please select a node id to update, available node ids are: " + str(node_id_list) + ": (if emtpy, the first one will be selected by default) ")
    if not node_id_to_update or not node_id_to_update.strip():
        node_id_to_update = node_id_list[0]
    
    assert node_id_to_update in node_id_list, f"The node id '{node_id_to_update}' is not valid."

    idle_container_cpu_memory_usage = input("Please enter the container's CPU memory usage in string (e.g., 300KB, 70MB, 1.5GB ...): ")
    idle_container_device_memory_usage = input("Please enter the container's device (accelerator device such as GPU, FPGA ...) memory usage in string (e.g., 300KB, 70MB, 1.5GB ...): ")

    # update the profile
    for profile in service_data["profiles"]:
        if profile["node_id"] == node_id_to_update:
            profile["idle_container_cpu_memory_usage"] = idle_container_cpu_memory_usage
            profile["idle_container_device_memory_usage"] = idle_container_device_memory_usage
            break
    # write the updated profile back to the json file
    with open(service_data_json_path, "w") as f:
        json.dump(service_data, f, indent=4)


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
