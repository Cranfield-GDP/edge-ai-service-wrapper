import os
import shutil
from dotenv import load_dotenv
from utils import (
    AI_CLIENT_SCRIPT_NAME,
    AI_SERVER_SCRIPT_NAME,
    AI_SERVER_UTILS_SCRIPT_NAME,
    HEALTHCHECK_SCRIPT_NAME,
    NECESSARY_SERVICE_FILE_LIST,
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


def code_generation_main(huggingface_model_name: str) -> None:

    # validate the hugging face model name
    if not validate_hf_model_name(huggingface_model_name):
        raise (f"The model '{huggingface_model_name}' is invalid.")
    print(f"The model {huggingface_model_name} is valid.")
    huggingface_model_readme = get_hf_model_readme(huggingface_model_name)
    if not huggingface_model_readme:
        raise (f"The model '{huggingface_model_name}' does not have a README file.")

    example_model_name = "example/image-model"
    example_model_directory = get_hf_model_directory(example_model_name)
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

    # --------------------------------
    # prepare the model output directory
    # --------------------------------
    hf_model_directory = get_hf_model_directory(huggingface_model_name)
    # reset the model directory if it exists
    if os.path.exists(hf_model_directory):
        shutil.rmtree(hf_model_directory)
    # create the model directory
    os.makedirs(hf_model_directory, exist_ok=True)
    print(f"The model directory {hf_model_directory} is prepared.")

    output_files_content = {}

    # --------------------------------
    # Copy the healthcheck.py file directly
    # --------------------------------
    copy_file_from_example_model_folder(
        HEALTHCHECK_SCRIPT_NAME,
        hf_model_directory, example_model_files_content, output_files_content
    )
    print(f"Healthcheck file copied to {hf_model_directory}.")

    # --------------------------------
    # Copy the ai_server.py file directly
    # --------------------------------
    copy_file_from_example_model_folder(
        AI_SERVER_SCRIPT_NAME,
        hf_model_directory, example_model_files_content, output_files_content
    )
    print(f"AI server file copied to {hf_model_directory}.")

    # --------------------------------
    # Copy the ai_server_utils.py file directly
    # --------------------------------
    copy_file_from_example_model_folder(
        AI_SERVER_UTILS_SCRIPT_NAME,
        hf_model_directory, example_model_files_content, output_files_content
    )
    print(f"AI server utils file copied to {hf_model_directory}.")

    # --------------------------------
    # Copy the ai_client.py file directly
    # --------------------------------
    copy_file_from_example_model_folder(
        AI_CLIENT_SCRIPT_NAME,
        hf_model_directory, example_model_files_content, output_files_content
    )
    print(f"AI client script copied to {hf_model_directory}.")

    # --------------------------------
    # Download the README file
    # --------------------------------
    download_model_readme(huggingface_model_name)
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
    )
    print(f"AI model script generated and saved.")
    
    # --------------------------------
    # Generate the xai_model.py if applicable
    # ---------------------------------
    model_task = get_model_pipeline_tag(huggingface_model_name)
    # currently only image-classification is supported for XAI
    xai_enabled = model_task == "image-classification"
    if xai_enabled:
        copy_file_from_example_model_folder(
            XAI_MODEL_SCRIPT_NAME,
            hf_model_directory, example_model_files_content, output_files_content
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
    )
    print(f"AI client utils script generated and saved.")


if __name__ == "__main__":
    huggingface_model_name = input(
        "Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): "
    )
    code_generation_main(huggingface_model_name)
