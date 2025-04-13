import json

def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request."""
    data = {}
    # Prompt the user to input the table query
    data["query"] = input("Please input the table query: ")

    # Prompt the user to specify the path to the JSON file containing the table data
    table_data_path = input("Please input the local JSON file path containing the table data: ")
    
    # Load the table data from the specified JSON file
    try:
        with open(table_data_path, 'r') as f:
            table_data = json.load(f)
    except Exception as e:
        print(f"Error loading table data from {table_data_path}: {e}")
        return None

    # Include the table data in the request data
    data["table_data"] = table_data
    
    return data

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request."""
    files = {}
    return files