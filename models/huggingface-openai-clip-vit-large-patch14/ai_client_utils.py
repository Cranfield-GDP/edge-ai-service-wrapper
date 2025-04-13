def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        files["file"] = image_file.read()
    return files

def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request including text prompts and unique execution ID."""
    data = {}

    # Allow the user to input a non-empty list of text prompts
    prompts = []
    print("Please input text prompts (type 'done' when finished):")
    while True:
        prompt = input("- ")
        if prompt.lower() == 'done':
            break
        if prompt.strip():
            prompts.append(prompt)

    if not prompts:
        raise ValueError("The list of text prompts cannot be empty.")

    data["text_prompts"] = prompts
    return data