from uuid import uuid4

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        files["file"] = image_file.read()
    return files

def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request."""
    data = {}
    # Generate a unique execution ID
    data["ue_id"] = str(uuid4())
    return data