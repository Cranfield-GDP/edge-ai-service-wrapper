import requests

# Server URL
SERVER_URL = "http://localhost:8000"  # Replace with the actual server URL if different

def send_image(file_path):
    """Send an image to the server and get predictions."""
    url = f"{SERVER_URL}/run"
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print("Predictions:")
        for prediction in response.json().get("predictions", []):
            print(f"Category ID: {prediction['category_id']}, Label: {prediction['label']}, Probability: {prediction['probability']:.4f}")
        print("Execution Time:", response.json().get("execution_time", "N/A"))
    else:
        print(f"Error: {response.status_code}, {response.text}")

def get_help():
    """Get help information from the server."""
    url = f"{SERVER_URL}/help"
    response = requests.get(url)
    if response.status_code == 200:
        print("Help Information:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}, {response.text}")

def get_resource_usage():
    """Get resource usage from the server."""
    url = f"{SERVER_URL}/resource"
    response = requests.get(url)
    if response.status_code == 200:
        print("Resource Usage:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    print("1. Send an image for prediction")
    print("2. Get help information")
    print("3. Get resource usage")
    choice = input("Enter your choice: ")

    if choice == "1":
        file_path = input("Enter the path to the image file: ")
        send_image(file_path)
    elif choice == "2":
        get_help()
    elif choice == "3":
        get_resource_usage()
    else:
        print("Invalid choice.")