# import any necessary libraries.
import requests
import uuid

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        files["file"] = image_file.read()
    return files

def prepare_ai_service_request_data():
    """Prepare the `data` part other than `ue_id` for the AI service request."""
    data = {}
    return data

def send_request_to_ai_service(url, files, data):
    """Send a request to the AI service endpoint."""
    response = requests.post(url, files=files, data=data)
    return response.json()

# Example usage
if __name__ == "__main__":
    url = "http://localhost:8000/run"  # Replace with the actual URL of the AI service
    files = prepare_ai_service_request_files()
    data = prepare_ai_service_request_data()
    result = send_request_to_ai_service(url, files, data)
    print("Response from AI service:", result)