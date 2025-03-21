import requests
import time

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
            print(f"Category ID: {prediction['category_id']}, Label: {prediction.get('category_name', 'N/A')}, Probability: {prediction['probability']:.4f}")
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

def pressure_test(file_path, num_requests):
    """Perform a pressure test by sending the image multiple times."""
    url = f"{SERVER_URL}/run"
    with open(file_path, "rb") as file:
        files = {"file": file.read()}  # Read the file once to reuse in all requests

    start_time = time.time()
    for i in range(num_requests):
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print(f"Request {i + 1}/{num_requests} succeeded.")
        else:
            print(f"Request {i + 1}/{num_requests} failed: {response.status_code}, {response.text}")
    total_time = time.time() - start_time
    print(f"Completed {num_requests} requests in {total_time:.2f} seconds, average time per request: {total_time / num_requests:.2f} seconds.")

if __name__ == "__main__":
    print("1. Send an image for prediction")
    print("2. Get help information")
    print("3. Get resource usage")
    print("4. Perform a pressure test")
    choice = input("Enter your choice: ")

    if choice == "1":
        file_path = input("Enter the path to the image file: ")
        send_image(file_path)
    elif choice == "2":
        get_help()
    elif choice == "3":
        get_resource_usage()
    elif choice == "4":
        file_path = input("Enter the path to the image file: ")
        num_requests = int(input("Enter the number of requests to send: "))
        pressure_test(file_path, num_requests)
    else:
        print("Invalid choice.")