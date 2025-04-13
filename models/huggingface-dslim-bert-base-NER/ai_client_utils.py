def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request."""
    data = {}
    # Prompt the user to input the text
    data["text"] = input("Please input the text to perform NER: ")
    data["ue_id"] = input("Please input the unique execution ID (ue_id): ")
    return data

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    return files