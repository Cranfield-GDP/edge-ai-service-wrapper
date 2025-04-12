import os
import shutil
from dotenv import load_dotenv
from model_specific_utils.yolov8_utils import YOLOv8_MODEL_ID_KEY, YOLOv8_MODEL_ID_LIST
from utils import (
    AI_CLIENT_SCRIPT_NAME,
    AI_SERVER_SCRIPT_NAME,
    AI_SERVER_UTILS_SCRIPT_NAME,
    NECESSARY_SERVICE_FILE_LIST,
    HEALTHCHECK_SCRIPT_NAME,
    XAI_MODEL_SCRIPT_NAME,
    copy_file_from_example_model_folder,
    download_model_readme,
    generate_ai_client_utils_script,
    generate_model_script,
    generate_dockerfile,
    get_hf_model_directory,
    get_hf_model_readme,
    get_model_pipeline_tag,
    validate_hf_model_name,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

def prepare_example_model_data():
    example_model_name = input(
        "Enter the example model name (default to microsoft/resnet-50): "
    ).strip()

    if not example_model_name:
        example_model_name = "microsoft/resnet-50"

    # handle special cases
    additional_data = {}
    if example_model_name == "Ultralytics/YOLOv8":
        while True:
            # ask for specific model_id inside the YOLOv8 repo
            if YOLOv8_MODEL_ID_KEY not in additional_data:
                print(f"Available YOLOv8 model IDs: {YOLOv8_MODEL_ID_LIST}")
                yolov8_model_id = input(
                    f"Enter the YOLOv8 model ID (default: {YOLOv8_MODEL_ID_LIST[0]}): "
                ).strip()
                if not yolov8_model_id:
                    yolov8_model_id = YOLOv8_MODEL_ID_LIST[0]
            else:
                print(f"Available YOLOv8 model IDs: {YOLOv8_MODEL_ID_LIST}")
                yolov8_model_id = input(
                    f"Enter the YOLOv8 model ID (default: {additional_data[YOLOv8_MODEL_ID_KEY]}): "
                ).strip()
                if not yolov8_model_id:
                    yolov8_model_id = additional_data[YOLOv8_MODEL_ID_KEY]

            if not yolov8_model_id or yolov8_model_id not in YOLOv8_MODEL_ID_LIST:
                print(
                    f"The model ID '{yolov8_model_id}' is invalid. Please choose from the available options."
                )
                continue

            additional_data[YOLOv8_MODEL_ID_KEY] = yolov8_model_id
            break

    example_model_directory = get_hf_model_directory(
        example_model_name, additional_data
    )

    assert os.path.exists(
        example_model_directory
    ), f"The example model directory '{example_model_directory}' does not exist."

    example_model_files_content = {}
    for file_name in NECESSARY_SERVICE_FILE_LIST:
        file_path = os.path.join(example_model_directory, file_name)
        assert os.path.exists(
            file_path
        ), f"The example file '{file_path}' does not exist."
        with open(file_path, "r") as file:
            example_model_files_content[file_name] = file.read()
    print(f"The example {example_model_name} is valid.")
    
    return example_model_files_content


def code_generation_main(huggingface_model_name: str, additional_data: dict) -> None:

    # validate the hugging face model name
    if not validate_hf_model_name(huggingface_model_name):
        raise (f"The model '{huggingface_model_name}' is invalid.")
    print(f"The model {huggingface_model_name} is valid.")
    huggingface_model_readme = get_hf_model_readme(huggingface_model_name)
    if not huggingface_model_readme:
        raise (f"The model '{huggingface_model_name}' does not have a README file.")

    example_model_files_content = prepare_example_model_data()

    # --------------------------------
    # prepare the model output directory
    # --------------------------------
    hf_model_directory = get_hf_model_directory(huggingface_model_name, additional_data)
    # reset the model directory if it exists
    if os.path.exists(hf_model_directory):
        shutil.rmtree(hf_model_directory)
    # create the model directory
    os.makedirs(hf_model_directory, exist_ok=True)
    print(f"The model directory {hf_model_directory} is prepared.")

    output_files_content = {}

    # --------------------------------
    #  Copy common scripts
    # -------------------------------
    for file_name_to_copy in [
        HEALTHCHECK_SCRIPT_NAME,
        AI_SERVER_SCRIPT_NAME,
        AI_SERVER_UTILS_SCRIPT_NAME,
        AI_CLIENT_SCRIPT_NAME,
    ]:
        copy_file_from_example_model_folder(
            file_name_to_copy,
            hf_model_directory,
            example_model_files_content,
            output_files_content,
        )
        print(f"File {file_name_to_copy} copied to {hf_model_directory}.")

    # --------------------------------
    # Download the README file
    # --------------------------------
    download_model_readme(huggingface_model_name, additional_data)
    print(f"README file downloaded to {hf_model_directory}.")

    # --------------------------------
    # Generating the model.py file
    # --------------------------------
    generate_model_script(
        huggingface_model_name,
        huggingface_model_readme,
        example_model_files_content,
        output_files_content,
        hf_model_directory,
        additional_data
    )
    print(f"AI model script generated and saved.")

    # --------------------------------
    # copy the xai_model.py if applicable
    # ---------------------------------
    model_task = get_model_pipeline_tag(huggingface_model_name)
    # currently only image-classification is supported for XAI
    xai_enabled = model_task == "image-classification"
    # placeholder for other checks in the future
    if xai_enabled:
        copy_file_from_example_model_folder(
            XAI_MODEL_SCRIPT_NAME,
            hf_model_directory,
            example_model_files_content,
            output_files_content,
        )
        print(f"XAI model script copied.")
    else:
        print(f"The model {huggingface_model_name} does not support XAI.")

    # --------------------------------
    # Generating the Dockerfile
    # --------------------------------
    generate_dockerfile(
        huggingface_model_name,
        huggingface_model_readme,
        example_model_files_content,
        output_files_content,
        hf_model_directory,
        additional_data,
    )
    print(f"Dockerfile generated and saved.")

    # --------------------------------
    # Generating the client_utils file
    # --------------------------------
    generate_ai_client_utils_script(
        huggingface_model_name,
        huggingface_model_readme,
        example_model_files_content,
        output_files_content,
        hf_model_directory,
        additional_data
    )
    print(f"AI client utils script generated and saved.")


if __name__ == "__main__":
    huggingface_model_name = input(
        "Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): "
    )
    code_generation_main(huggingface_model_name)
