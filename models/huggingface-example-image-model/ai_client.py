import json
import requests
import time

# import any other necessary libraries

SERVER_URL = input(
    "Please input server URL (e.g., http://localhost:54660): "
)  # Replace with the actual server URL if different


def send_post_request(url, files, data):
    """Send request to run AI service and display AI service responses."""

    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def run_inferece_test(ue_id):
    url = f"{SERVER_URL}/model/run"

    files = {}  # if files are required, use input to ask for the file path

    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        # Ensure the file is opened in binary mode
        files["file"] = image_file
        data = {"ue_id": ue_id}
        print(send_post_request(url, files, data))


def run_pressure_test(ue_id, num_requests):
    url = f"{SERVER_URL}/model/profile"

    files = {}  # if files are required, use input to ask for the file path

    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        files["file"] = image_file.read()  # Read the file since we are sending multiple times.

    # Ensure the file is opened in binary mode
    data = {"ue_id": ue_id}

    start_time = time.time()
    try:
        average_profile_result = {
            "name": None,
            "device_type": None,
            "device_name": None,
            "cpu_memory_usage": 0,
            "self_cpu_memory_usage": 0,
            "device_memory_usage": 0,
            "self_device_memory_usage": 0,

            "cpu_time_total": 0,
            "self_cpu_time_total": 0,
            "device_time_total": 0,
            "self_device_time_total": 0,
        }

        for _ in range(num_requests):
            profile_response = send_post_request(url, files, data)
            print(json.dumps(profile_response, indent=4))
            if profile_response:
                # update the name, device type and device name
                if average_profile_result["name"] is None:
                    average_profile_result["name"] = profile_response["profile_result"]["name"]
                    average_profile_result["device_type"] = profile_response["profile_result"]["device_type"]
                    average_profile_result["device_name"] = profile_response["profile_result"]["device_name"]
                # update the average profile result
                average_profile_result["cpu_memory_usage"] += profile_response["profile_result"]["cpu_memory_usage"]
                average_profile_result["self_cpu_memory_usage"] += profile_response["profile_result"]["self_cpu_memory_usage"]
                average_profile_result["device_memory_usage"] += profile_response["profile_result"]["device_memory_usage"]
                average_profile_result["self_device_memory_usage"] += profile_response["profile_result"]["self_device_memory_usage"]
                average_profile_result["cpu_time_total"] += profile_response["profile_result"]["cpu_time_total"]
                average_profile_result["self_cpu_time_total"] += profile_response["profile_result"]["self_cpu_time_total"]
                average_profile_result["device_time_total"] += profile_response["profile_result"]["device_time_total"]
                average_profile_result["self_device_time_total"] += profile_response["profile_result"]["self_device_time_total"]

        # Calculate the average values
        average_profile_result["cpu_memory_usage"] /= num_requests
        average_profile_result["self_cpu_memory_usage"] /= num_requests
        average_profile_result["device_memory_usage"] /= num_requests
        average_profile_result["self_device_memory_usage"] /= num_requests
        average_profile_result["cpu_time_total"] /= num_requests
        average_profile_result["self_cpu_time_total"] /= num_requests
        average_profile_result["device_time_total"] /= num_requests
        average_profile_result["self_device_time_total"] /= num_requests

        print("Average Profile Result:")
        print(f"Name: {average_profile_result['name']}")
        print(f"Device Type: {average_profile_result['device_type']}")
        print(f"Device Name: {average_profile_result['device_name']}")
        # for memory convert bytes into MB
        print(f"CPU Memory Usage: {average_profile_result['cpu_memory_usage'] / (1024 * 1024):.2f} MB")
        print(f"Self CPU Memory Usage: {average_profile_result['self_cpu_memory_usage'] / (1024 * 1024):.2f} MB")
        print(f"Device Memory Usage: {average_profile_result['device_memory_usage'] / (1024 * 1024):.2f} MB")
        print(f"Self Device Memory Usage: {average_profile_result['self_device_memory_usage'] / (1024 * 1024):.2f} MB")
        # for time convert microseconds into milliseconds
        print(f"CPU Time Total: {average_profile_result['cpu_time_total'] / 1000:.2f} ms")
        print(f"Self CPU Time Total: {average_profile_result['self_cpu_time_total'] / 1000:.2f} ms")
        print(f"Device Time Total: {average_profile_result['device_time_total'] / 1000:.2f} ms")
        print(f"Self Device Time Total: {average_profile_result['self_device_time_total'] / 1000:.2f} ms")
        
        total_time = time.time() - start_time
        print(
            f"Completed {num_requests} requests in {total_time:.2f} seconds, average time per request: {total_time / num_requests:.2f} seconds."
        )
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


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


if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Run AI service once")
        print("2. Get help information")
        print("3. Get logs for a specific UE_ID")
        print(
            "4. Perform a pressure test to profile the runtime statistics the AI service"
        )
        print("5. Quit")
        choice = input("Enter your choice: ")

        if choice == "1":
            ue_id = input("Enter the UE_ID: ")
            run_inferece_test(ue_id)
        elif choice == "2":
            get_help()
        elif choice == "3":
            ue_id = input("Enter the UE_ID: ")
            get_ue_log(ue_id)
        elif choice == "4":
            ue_id = input("Enter the UE_ID: ")
            num_requests = int(input("Enter the number of requests to send: "))
            run_pressure_test(ue_id, num_requests)
        elif choice == "5":
            print("Exiting the client. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
