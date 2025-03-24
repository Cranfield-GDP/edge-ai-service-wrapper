import subprocess
import sys
from code_generation import code_generation_main
from docker_validation import build_and_start_docker_container
from docker_image_upload import push_docker_image_main
import traceback
from utils import (
    get_hf_model_directory,
    download_model_readme,
    copy_test_image,
    update_edge_ai_service_db,
)

def main():
    huggingface_model_name_default = "microsoft/resnet-50"  # Example model name
    huggingface_model_name = huggingface_model_name_default

    while True:
        print("\nSelect an option:")
        print("1. Generate codes to wrap a given AI model into a FastAPI service")
        print("2. Build and run the docker container for the AI service")
        print("3. Push the docker image to the Docker Hub")
        print("4. Open another terminal to test the AI service")
        print("5. Download the README file for the Hugging Face model")
        print("6. Copy a test image for the Hugging Face model")
        print("7. Update the edge AI service database")
        print("q. Quit")
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == "1":
                new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
                if new_model_name.strip():
                    huggingface_model_name = new_model_name
                code_generation_main(huggingface_model_name)
                print("Code generation completed successfully.")
            
            elif choice == "2":
                new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
                if new_model_name.strip():
                    huggingface_model_name = new_model_name
                build_and_start_docker_container(huggingface_model_name)
                print("Docker validation and container startup completed successfully.")
            
            elif choice == "3":
                new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
                if new_model_name.strip():
                    huggingface_model_name = new_model_name
                push_docker_image_main(huggingface_model_name)
                print("Docker image upload completed successfully.")
            
            elif choice == "4":
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

            elif choice == "5":
                new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
                if new_model_name.strip():
                    huggingface_model_name = new_model_name
                try:
                    readme_path = download_model_readme(huggingface_model_name)
                    print(f"README downloaded successfully: {readme_path}")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == "6":
                new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
                if new_model_name.strip():
                    huggingface_model_name = new_model_name
                try:
                    test_image_path = copy_test_image(huggingface_model_name)
                    print(f"Test image copied successfully: {test_image_path}")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == "7":
                new_model_name = input(f"Enter the Hugging Face model name (default: {huggingface_model_name}): ")
                if new_model_name.strip():
                    huggingface_model_name = new_model_name
                try:
                    update_edge_ai_service_db(huggingface_model_name)
                    print("Edge AI service database updated successfully.")
                except Exception as e:
                    print(f"Error: {e}")

            elif choice == "q":
                print("Exiting the program. Goodbye!")
                sys.exit(0)
            
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