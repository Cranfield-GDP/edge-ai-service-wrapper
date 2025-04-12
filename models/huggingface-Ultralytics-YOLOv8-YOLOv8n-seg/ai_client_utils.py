# import any necessary libraries.

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        files["file"] = image_file.read()
    return files

def prepare_ai_service_request_data():
    """Prepare the `data` part including `ue_id` for the AI service request."""
    data = {}
    ue_id = input("Please input the User Equipment ID (ue_id): ")
    data["ue_id"] = ue_id
    return data