def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    audio_file_path = input("Please input the audio file path: ")
    with open(audio_file_path, "rb") as audio_file:
        files["file"] = audio_file.read()
    return files

def prepare_ai_service_request_data():
    """Prepare the `data` part other than `ue_id` for the AI service request."""
    data = {}
    return data