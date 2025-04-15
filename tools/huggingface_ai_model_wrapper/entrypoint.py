import sys
from rich.console import Console
from rich.table import Table
import subprocess
from code_generation import code_generation_main
from docker_validation import build_ai_service_base_image, build_and_start_docker_container, update_container_memory_usage, stop_docker_container
from docker_image_upload import push_docker_image_main
import traceback
from model_specific_utils.yolov8_utils import YOLOv8_MODEL_ID_KEY, YOLOv8_MODEL_ID_LIST
from utils import (
    get_hf_model_directory,
    copy_test_file,
    update_ai_service_db,
    prepare_service_data_json,
)
from management import sync_code_with_image_and_database

huggingface_model_name = "microsoft/resnet-50"
additional_data = {}

def prompt_for_model_name():
    global huggingface_model_name, additional_data

    new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
    if new_model_name.strip():
        huggingface_model_name = new_model_name
    
    # handle special cases
    if huggingface_model_name == "Ultralytics/YOLOv8":
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


def option_build_service_base_image():
    """Build the base image for the AI service."""
    build_ai_service_base_image()
    print("Base image build completed successfully.")

def option_code_generation():
    """Generate codes to wrap a given AI model into a FastAPI service."""
    global huggingface_model_name, additional_data
    prompt_for_model_name()
    code_generation_main(huggingface_model_name, additional_data)
    print("Code generation completed successfully.")

def option_prepare_model_service_data():
    """Prepare model service data json."""
    global huggingface_model_name, additional_data
    prompt_for_model_name()
    prepare_service_data_json(huggingface_model_name, additional_data)
    print(f"service_data.json prepared succesfully.")

def option_build_and_run_docker_container():
    """Build and run the docker container for the AI service."""
    global huggingface_model_name, additional_data
    prompt_for_model_name()
    build_and_start_docker_container(huggingface_model_name, additional_data)
    print("Docker validation and container startup completed successfully.")

def option_push_docker_image():
    """Push the docker image to the Docker Hub."""
    global huggingface_model_name, additional_data
    prompt_for_model_name()
    push_docker_image_main(huggingface_model_name, additional_data)
    print("Docker image upload completed successfully.")

def option_open_client_script_terminal():
    """Open another terminal to test the AI service."""
    global huggingface_model_name, additional_data

    print("Opening a new terminal to test the AI service...")
    print("Please run the client script in the new terminal.")
    print("By default, it uses the same python executable as this script.")
    print("If you want to use a different python executable, please specify it in the client script.")

    python_executable = sys.executable
    
    prompt_for_model_name()

    ai_model_directory = get_hf_model_directory(huggingface_model_name, additional_data)

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

def option_copy_test_file():
    """Copy a test file for the Hugging Face model."""
    global huggingface_model_name, additional_data
    prompt_for_model_name()

    try:
        test_file_path = copy_test_file(huggingface_model_name, additional_data)
        print(f"Test file copied successfully: {test_file_path}")
    except Exception as e:
        print(f"Error: {e}")

def option_update_ai_service_database():
    """Update the edge AI service database."""
    global huggingface_model_name, additional_data
    prompt_for_model_name()
    try:
        update_ai_service_db(huggingface_model_name, additional_data)
        print("Edge AI service database updated successfully.")
    except Exception as e:
        print(f"Error: {e}")


def option_update_container_memory_usage():
    """Manually update the container cpu memory and device memory usage."""
    global huggingface_model_name, additional_data
    prompt_for_model_name()
    try:
        update_container_memory_usage(huggingface_model_name, additional_data)
        print("Container memory usage updated successfully.")
    except Exception as e:
        print(f"Error: {e}")

def option_stop_docker_container():
    """Stop the docker container."""
    global huggingface_model_name, additional_data
    prompt_for_model_name()
    try:
        stop_docker_container(huggingface_model_name, additional_data)
        print("Docker container stopped successfully.")
    except Exception as e:
        print(f"Error: {e}")


def option_sync_code_with_image_and_database():
    """Sync the AI service source code with AI service images and database."""
    try:
        sync_code_with_image_and_database()
        print("Syncing code with image and database...")
    except Exception as e:
        print(f"Error: {e}")


OPTIONS = {
    "Management Options": [
        {
            "label": "Build the base image for the AI service",
            "function": option_build_service_base_image,
        },
        {
            "label": "Sync the AI service source code with AI service images and database",
            "function": option_sync_code_with_image_and_database,
        }
    ],
    "AI Service Options": [
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
            "label": "Copy a test file to the model directory",
            "function": option_copy_test_file,
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
        {
            "label": "Stop the docker container",
            "function": option_stop_docker_container,
        },
    ],
}

def main():
    console = Console()

    while True:
        # Create a Rich table for the options
        table = Table(title="AI Service Options")
        table.add_column("Option", justify="center", style="cyan", no_wrap=True)
        table.add_column("Description", style="yellow")

        option_index = 1
        for section, options in OPTIONS.items():
            # Add section header
            table.add_row("", f"[bold magenta]{section}[/bold magenta]")
            table.add_row("", "[dim]-----------------------------[/dim]")

            for option in options:
                table.add_row(str(option_index), option["label"])
                option_index += 1

            # Add a separator row for better readability
            table.add_row("", "[dim]-----------------------------[/dim]")

        # Add the Quit option at the end
        table.add_row("q", "Quit")

        # Print the table
        console.print(table)

        try:
            choice = input("Enter your choice (1-7 or 'q' to quit): ").strip()
            
            if choice == "q":
                console.print("[bold green]Exiting the program. Goodbye![/bold green]")
                sys.exit(0)
            elif choice.isdigit() and 1 <= int(choice) <= sum(len(options) for options in OPTIONS.values()):
                # Find the selected option
                option_index = int(choice)
                for options in OPTIONS.values():
                    if option_index <= len(options):
                        selected_option = options[option_index - 1]
                        break
                    option_index -= len(options)
                
                console.print(f"[bold yellow]Executing:[/bold yellow] {selected_option['label']}")
                selected_option["function"]()
            else:
                console.print("[bold red]Invalid choice. Please select a valid option (1-7 or 'q').[/bold red]")
        
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {e}")
            traceback.print_exc()
            console.print("[bold red]Returning to the main menu...[/bold red]")

if __name__ == "__main__":
    main()