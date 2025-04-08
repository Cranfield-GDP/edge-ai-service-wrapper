import sys
import subprocess
from code_generation import code_generation_main
from docker_validation import build_and_start_docker_container, update_container_memory_usage
from docker_image_upload import push_docker_image_main
import traceback
from utils import (
    get_hf_model_directory,
    copy_test_image,
    update_ai_service_db,
    prepare_service_data_json,
)

huggingface_model_name_default = "microsoft/resnet-50"  # Example model name
huggingface_model_name = huggingface_model_name_default

def option_code_generation():
    """Generate codes to wrap a given AI model into a FastAPI service."""
    global huggingface_model_name

    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name
    code_generation_main(huggingface_model_name)
    print("Code generation completed successfully.")

def option_prepare_model_service_data():
    """Prepare model service data json."""
    global huggingface_model_name

    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name

    prepare_service_data_json(huggingface_model_name)
    print(f"service_data.json prepared succesfully.")

def option_build_and_run_docker_container():
    """Build and run the docker container for the AI service."""
    global huggingface_model_name

    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name
    build_and_start_docker_container(huggingface_model_name)
    print("Docker validation and container startup completed successfully.")

def option_push_docker_image():
    """Push the docker image to the Docker Hub."""
    global huggingface_model_name

    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name
    push_docker_image_main(huggingface_model_name)
    print("Docker image upload completed successfully.")

def option_open_client_script_terminal():
    """Open another terminal to test the AI service."""
    global huggingface_model_name

    print("Opening a new terminal to test the AI service...")
    print("Please run the client script in the new terminal.")
    print("By default, it uses the same python executable as this script.")
    print("If you want to use a different python executable, please specify it in the client script.")

    python_executable = sys.executable
    
    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name
    ai_model_directory = get_hf_model_directory(huggingface_model_name)

    # Check if system is Windows or Linux/Mac
    # Start a new terminal, change to folder to the model directory, and run the client script
    if sys.platform.startswith("win"):
        subprocess.run(
            ["start", "cmd", "/k", f"cd {ai_model_directory} && {python_executable} ai_client.py"],
            shell=True,
        )
    else:
        subprocess.run(
            ["gnome-terminal", "--", "bash", "-c", f"cd {ai_model_directory} && {python_executable} ai_client.py; exec bash"],
            shell=True,
        )

def option_copy_test_image():
    """Copy a test image for the Hugging Face model."""
    global huggingface_model_name

    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name
    try:
        test_image_path = copy_test_image(huggingface_model_name)
        print(f"Test image copied successfully: {test_image_path}")
    except Exception as e:
        print(f"Error: {e}")

def option_update_ai_service_database():
    """Update the edge AI service database."""
    global huggingface_model_name
    
    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name
    try:
        update_ai_service_db(huggingface_model_name)
        print("Edge AI service database updated successfully.")
    except Exception as e:
        print(f"Error: {e}")


def option_update_container_memory_usage():
    """Manually update the container cpu memory and device memory usage."""

    global huggingface_model_name

    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name
    try:
        update_container_memory_usage(huggingface_model_name)
        print("Container memory usage updated successfully.")
    except Exception as e:
        print(f"Error: {e}")

OPTIONS = [
    {
        "label": "Generate codes to wrap a given AI model into a FastAPI service",
        "function": option_code_generation,
    },
    {
        "label": "Prepare model service data json",
        "function": option_prepare_model_service_data,
    },
    {
        "label": "Build and run the docker container for the AI service",
        "function": option_build_and_run_docker_container,
    },
    {
        "label": "Copy a test image to the model directory",
        "function": option_copy_test_image,
    },
    {
        "label": "Open another terminal to test the AI service",
        "function": option_open_client_script_terminal,
    },
    {
        "label": "Update container memory usage",
        "function": option_update_container_memory_usage,
    },
    {
        "label": "Push the docker image to the Docker Hub",
        "function": option_push_docker_image,
    },
    {
        "label": "Update the AI service database",
        "function": option_update_ai_service_database,
    },
]


def main():
    while True:
        print("\nSelect an option:")
        for i, option in enumerate(OPTIONS, start=1):
            print(f"{i}. {option['label']}")
        print("q. Quit")
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == "q":
                print("Exiting the program. Goodbye!")
                sys.exit(0)
            elif choice.isdigit() and 1 <= int(choice) <= len(OPTIONS):
                option = OPTIONS[int(choice) - 1]
                print(f"Executing: {option['label']}")
                option["function"]()
            else:
                print("Invalid choice. Please select a valid option (1-7).")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            # print the stack trace for debugging
            traceback.print_exc()
            print("Please try again.")
            print("Returning to the main menu...")

if __name__ == "__main__":
    main()