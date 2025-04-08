import requests

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}  # if files are required, use input to ask for the file path
    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        # save the content of the file instead of a buffered reader
        # to avoid "read from closed file" error
        files["file"] = image_file.read()
    return files

def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request."""
    data = {
        "ue_id": input("Please input the ue_id: ")
    }
    return data

def test_image_classification_service(base_url):
    endpoint = f"{base_url}/run"
    
    files = prepare_ai_service_request_files()
    data = prepare_ai_service_request_data()
    
    response = requests.post(endpoint, files=files, data=data)
    
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Failed:", response.status_code, response.text)

# Usage
# base_url = "http://localhost:8000"
# test_image_classification_service(base_url)