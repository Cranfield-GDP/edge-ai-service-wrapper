import requests
import time

# Server URL
SERVER_URL = "http://localhost:8001"  # Replace with the actual server URL if different

def send_image(file_path, ue_id):
    """Send an image to the server and get predictions."""
    url = f"{SERVER_URL}/run"
    with open(file_path, "rb") as file:
        files = {"file": file}
        data = {"ue_id": ue_id}
        print("Sending image to server...")
        response = requests.post(url, files=files, data=data, timeout=5)
    
    if response.status_code == 200:
        print("Predictions:")
        for prediction in response.json().get("predictions", []):
            print(f"Category ID: {prediction['category_id']}, Label: {prediction['label']}, Probability: {prediction['probability']:.4f}")
        print("Execution Duration:", response.json().get("execution_duration", "N/A"))
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

def get_ue_log(ue_id):
    """Retrieve logs for a specific UE_ID."""
    url = f"{SERVER_URL}/get_ue_log"
    params = {"ue_id": ue_id}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print(f"Logs for UE_ID {ue_id}:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}, {response.text}")

def pressure_test(file_path, ue_id, num_requests):
    """Perform a pressure test by sending the image multiple times."""
    url = f"{SERVER_URL}/run"
    with open(file_path, "rb") as file:
        files = {"file": file.read()}  # Read the file once to reuse in all requests

    start_time = time.time()
    for i in range(num_requests):
        response = requests.post(url, files={"file": files["file"]}, data={"ue_id": ue_id})
        if response.status_code == 200:
            print(f"Request {i + 1}/{num_requests} succeeded.")
        else:
            print(f"Request {i + 1}/{num_requests} failed: {response.status_code}, {response.text}")
    total_time = time.time() - start_time
    print(f"Completed {num_requests} requests in {total_time:.2f} seconds, average time per request: {total_time / num_requests:.2f} seconds.")

if __name__ == "__main__":
    print("1. Send an image for prediction")
    print("2. Get help information")
    print("3. Get logs for a specific UE_ID")
    print("4. Perform a pressure test")
    choice = input("Enter your choice: ")

    if choice == "1":
        file_path = input("Enter the path to the image file: ")
        ue_id = input("Enter the UE_ID: ")
        send_image(file_path, ue_id)
    elif choice == "2":
        get_help()
    elif choice == "3":
        ue_id = input("Enter the UE_ID: ")
        get_ue_log(ue_id)
    elif choice == "4":
        file_path = input("Enter the path to the image file: ")
        ue_id = input("Enter the UE_ID: ")
        num_requests = int(input("Enter the number of requests to send: "))
        pressure_test(file_path, ue_id, num_requests)
    else:
        print("Invalid choice.")