# import any necessary libraries.

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}  # if files are required, use input to ask for the file path
    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        # save the content of the file instead of a buffered reader
        files["file"] = image_file.read()
    return files


def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request."""
    data = {}
    # Add any additional data needed by the AI service
    return data
