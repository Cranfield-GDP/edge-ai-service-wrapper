import tkinter as tk
from tkinter import messagebox
from code_generation import code_generation_main
from docker_validation import build_and_start_docker_container
from docker_image_upload import push_docker_image_main
from utils import (
    get_docker_image_build_name,
    get_hf_model_directory,
    download_model_readme,
    copy_test_image,
)
import subprocess
import sys

huggingface_model_name_default = "microsoft/resnet-50"  # Example model name


def generate_code():
    model_name = model_name_entry.get().strip() or huggingface_model_name_default
    try:
        code_generation_main(model_name)
        messagebox.showinfo("Success", "Code generation completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def build_and_run_docker():
    model_name = model_name_entry.get().strip() or huggingface_model_name_default
    try:
        build_and_start_docker_container(model_name)
        messagebox.showinfo("Success", "Docker validation and container startup completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def push_docker_image():
    model_name = model_name_entry.get().strip() or huggingface_model_name_default
    try:
        image_name = get_docker_image_build_name(model_name)
        push_docker_image_main(image_name)
        messagebox.showinfo("Success", "Docker image upload completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def open_terminal():
    model_name = model_name_entry.get().strip() or huggingface_model_name_default
    try:
        model_directory = get_hf_model_directory(model_name)
        python_executable = sys.executable

        if sys.platform.startswith("win"):
            subprocess.run(
                ["start", "cmd", "/k", f"cd {model_directory} && {python_executable} ai_client.py"],
                shell=True,
            )
        else:
            subprocess.run(
                ["gnome-terminal", "--", "bash", "-c", f"cd {model_directory} && {python_executable} ai_client.py; exec bash"],
                shell=True,
            )
        messagebox.showinfo("Info", "Opened a new terminal to test the AI service.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def download_readme():
    model_name = model_name_entry.get().strip() or huggingface_model_name_default
    try:
        readme_path = download_model_readme(model_name)
        messagebox.showinfo("Success", f"README downloaded successfully: {readme_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def copy_test_image_ui():
    model_name = model_name_entry.get().strip() or huggingface_model_name_default
    try:
        image_path = copy_test_image(model_name)
        messagebox.showinfo("Success", f"Test image copied successfully: {image_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def quit_app():
    root.destroy()


# Create the main UI
root = tk.Tk()
root.title("Hugging Face AI Model Wrapper")

# Add a label and text entry for the Hugging Face model name
tk.Label(root, text="Enter Hugging Face Model Name:").pack(pady=5)
model_name_entry = tk.Entry(root, width=50)
model_name_entry.insert(0, huggingface_model_name_default)  # Set default value
model_name_entry.pack(pady=5)

# Add buttons for each option
tk.Button(root, text="1. Generate Codes", command=generate_code, width=40).pack(pady=5)
tk.Button(root, text="2. Build and Run Docker", command=build_and_run_docker, width=40).pack(pady=5)
tk.Button(root, text="3. Push Docker Image", command=push_docker_image, width=40).pack(pady=5)
tk.Button(root, text="4. Open Terminal to Test AI Service", command=open_terminal, width=40).pack(pady=5)
tk.Button(root, text="5. Download README", command=download_readme, width=40).pack(pady=5)
tk.Button(root, text="6. Copy Test Image", command=copy_test_image_ui, width=40).pack(pady=5)
tk.Button(root, text="7. Quit", command=quit_app, width=40).pack(pady=5)

# Run the UI loop
root.mainloop()