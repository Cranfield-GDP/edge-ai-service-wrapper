# import necessary libraries
import os

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    image_file_path = input("Please input the image file path: ")
    if not os.path.exists(image_file_path):
        raise FileNotFoundError(f"The image file {image_file_path} does not exist.")
    with open(image_file_path, "rb") as image_file:
        files["file"] = image_file.read()
    return files

def prepare_ai_service_request_data():
    """Prepare the `data` part including `ue_id` for the AI service request."""
    data = {}
    ue_id = input("Please input the unique execution ID (ue_id): ")
    data["ue_id"] = ue_id
    return data