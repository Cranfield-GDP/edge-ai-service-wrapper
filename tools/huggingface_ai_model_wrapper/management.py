import json
import os
import subprocess

from docker_image_upload import push_docker_image_main
from model_specific_utils.yolov8_utils import YOLOv8_MODEL_ID_KEY
from utils import SERVICE_DATA_JSON_NAME, get_docker_image_build_name, get_hf_model_directory, sync_code_content_to_service_data_json, update_ai_service_db


def sync_code_with_image_and_database() -> None:
    hf_models_directory = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
    )

    # get all the huggingface model directories
    example_model_directory_to_exclude = get_hf_model_directory(
        "example/image-model", None
    )

    # model_directories = [
    #     os.path.join(hf_models_directory, model_dir)
    #     for model_dir in os.listdir(hf_models_directory)
    #     if os.path.isdir(os.path.join(hf_models_directory, model_dir))
    #     and os.path.join(hf_models_directory, model_dir) != example_model_directory_to_exclude
    # ]

    for model_dir in os.listdir(hf_models_directory):
        model_dir_full = os.path.join(hf_models_directory, model_dir)
        if not os.path.isdir(model_dir_full):
            continue

        if model_dir_full == example_model_directory_to_exclude:
            continue

        service_data_json_path = os.path.join(model_dir_full, SERVICE_DATA_JSON_NAME)

        assert os.path.exists(
            service_data_json_path
        ), f"Service data json file not found: {service_data_json_path}"

        with open(service_data_json_path, "r") as file:
            service_data_json = json.load(file)
            # get the image repository url
            image_repository_url = service_data_json["image_repository_url"]

        model_name = service_data_json["model_name"]

        # handle special cases for YOLOv8 model series
        additional_data = {}
        if model_name == "Ultralytics/YOLOv8":
            yolov8_model_id = model_dir.replace(f"huggingface-Ultralytics-YOLOv8-", "")
            additional_data[YOLOv8_MODEL_ID_KEY] = yolov8_model_id

        # synchronize all the code content within service_data.json
        sync_code_content_to_service_data_json(service_data_json, model_dir_full)

        # save the service_data.json file
        with open(service_data_json_path, "w") as file:
            json.dump(service_data_json, file, indent=4)

        # build the docker image
        docker_image_build_name = get_docker_image_build_name(
            model_name, additional_data
        )
        subprocess.run(
            ["docker", "build", "-t", docker_image_build_name, "."],
            cwd=model_dir_full,
            check=True,
        )
        print(f"Docker image {docker_image_build_name} built successfully.")

        push_docker_image_main(model_name, additional_data)

        update_ai_service_db(model_name, additional_data)

        print("Docker image pushed to Docker Hub and database updated successfully.")