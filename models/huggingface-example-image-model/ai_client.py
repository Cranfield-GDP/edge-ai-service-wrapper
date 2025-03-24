import requests
import time
# import any other necessary libraries

SERVER_URL = input("Please input server URL (e.g., http://localhost:54660): ")  # Replace with the actual server URL if different

def call_ai_service(file_path, ue_id):
    """Send request to run AI service and display AI service responses."""
    pass

def get_help():
    """Get help information from the server."""
    url = f"{SERVER_URL}/help"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Help Information:")
            print(response.json())
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def get_ue_log(ue_id):
    """Retrieve logs for a specific UE_ID."""
    url = f"{SERVER_URL}/get_ue_log"
    params = {"ue_id": ue_id}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print(f"Logs for UE_ID {ue_id}:")
            print(response.json())
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def pressure_test(file_path, ue_id, num_requests):
    """Perform a pressure test by calling ai serivce multiple times."""
    url = f"{SERVER_URL}/run"
    with open(file_path, "rb") as file:
        files = {"file": file.read()}  # Read the file once to reuse in all requests

    start_time = time.time()
    try:
        for i in range(num_requests):
            call_ai_service(file_path, ue_id)
        total_time = time.time() - start_time
        print(f"Completed {num_requests} requests in {total_time:.2f} seconds, average time per request: {total_time / num_requests:.2f} seconds.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Run AI service once")
        print("2. Get help information")
        print("3. Get logs for a specific UE_ID")
        print("4. Perform a pressure test")
        print("5. Quit")
        choice = input("Enter your choice: ")

        if choice == "1":
            file_path = input("Enter the path to the image file: ")
            ue_id = input("Enter the UE_ID: ")
            call_ai_service(file_path, ue_id)
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
        elif choice == "5":
            print("Exiting the client. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")