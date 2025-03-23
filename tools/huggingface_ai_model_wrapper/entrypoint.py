import subprocess
import sys
from code_generation import code_generation_main
from docker_validation import build_and_start_docker_container
from docker_image_upload import push_docker_image_main
from utils import get_docker_image_build_name, get_hf_model_directory

def main():
    while True:
        print("\nSelect an option:")
        print("1. Generate codes to wrap a given AI model into a FastAPI service")
        print("2. Build and run the docker container for the AI service")
        print("3. Push the docker image to the Docker Hub")
        print("4. Open another terminal to test the AI service")
        print("5. Quit")
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                huggingface_model_name = input("Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): ")
                code_generation_main(huggingface_model_name)
                print("Code generation completed successfully.")
            
            elif choice == "2":
                huggingface_model_name = input("Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): ")
                build_and_start_docker_container(huggingface_model_name)
                print("Docker validation and container startup completed successfully.")
            
            elif choice == "3":
                huggingface_model_name = input("Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): ")
                ai_service_image_name = get_docker_image_build_name(huggingface_model_name)
                push_docker_image_main(ai_service_image_name)
                print("Docker image upload completed successfully.")
            
            elif choice == "4":
                print("Opening a new terminal to test the AI service...")
                print("Please run the client script in the new terminal.")
                print("By default, it uses the same python exectuable as this script.")
                print("If you want to use a different python executable, please specify it in the client script.")

                python_executable = sys.executable
                
                huggingface_model_name = input("Enter the Hugging Face model name (e.g., 'microsoft/resnet-50'): ")
                ai_model_directory = get_hf_model_directory(huggingface_model_name)

                # check if system is Windows or Linux/Mac
                # start a new terminal, change to folder to the model directory, and run the client script
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
                print("Exiting the program. Goodbye!")
                sys.exit(0)
            
            else:
                print("Invalid choice. Please select a valid option (1-4).")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Returning to the main menu...")

if __name__ == "__main__":
    main()