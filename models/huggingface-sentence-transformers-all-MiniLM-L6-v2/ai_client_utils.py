def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request."""
    data = {}
    sentences = input("Please input the sentences separated by a comma: ")
    data["sentences"] = [sentence.strip() for sentence in sentences.split(",")]
    return data

def prepare_ai_service_request_files():
    """There are no files to be uploaded for this service, return an empty dictionary."""
    return {}