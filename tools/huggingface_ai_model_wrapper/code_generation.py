import os
import shutil
from dotenv import load_dotenv
from utils import (
    NECESSARY_SERVICE_FILE_LIST,
    copy_ai_client_utils_script,
    copy_healthcheck_script,
    download_model_readme,
    generate_ai_client_script,
    generate_model_script,
    generate_dockerfile,
    get_hf_model_directory,
    get_hf_model_readme,
    validate_hf_model_name,
    copy_ai_server_script,
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
    copy_healthcheck_script(
        hf_model_directory, example_model_files_content, output_files_content
    )
    print(f"Healthcheck file copied to {hf_model_directory}.")

    # --------------------------------
    # Copy the ai_server.py file directly
    # --------------------------------
    copy_ai_server_script(
        hf_model_directory, example_model_files_content, output_files_content
    )
    print(f"AI server file copied to {hf_model_directory}.")

    # --------------------------------
    # Copy the ai_client_utils.py file directly
    # --------------------------------
    copy_ai_client_utils_script(
        hf_model_directory, example_model_files_content, output_files_content
    )
    print(f"AI client utils script copied to {hf_model_directory}.")

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
    # Generating the client file
    # --------------------------------
    generate_ai_client_script(
        huggingface_model_name,
        huggingface_model_readme,
        example_model_files_content,
        output_files_content,
        hf_model_directory,
    )
    print(f"AI client script generated and saved.")


if __name__ == "__main__":
    huggingface_model_name = input(
        "Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): "
    )
    code_generation_main(huggingface_model_name)
