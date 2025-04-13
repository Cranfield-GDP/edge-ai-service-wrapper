def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request."""
    data = {}
    # Prompt the user to input the text
    data["sequence"] = input("Please input the text to be classified: ")

    while True:
        data["candidate_labels"] = input("Please input the candidate labels (comma-separated): ").split(",")
        # Remove leading and trailing spaces from each label
        data["candidate_labels"] = [label.strip() for label in data["candidate_labels"]]
        # Check if the user provided any labels
        if not data["candidate_labels"]:
            print("No candidate labels provided. Please provide at least one label.")
            continue
        break
    return data

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    return files