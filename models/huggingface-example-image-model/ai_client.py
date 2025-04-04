import json
import requests
import time
from ai_client_utils import get_help, get_ue_log, ProfileResultProcessor, get_inference_url, get_profile_url
# import any other necessary libraries

# ask the user to input the server URL
SERVER_URL = input(
    "Please input server URL (e.g., http://localhost:54660): "
)

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
    files = {}  # if files are required, use input to ask for the file path

    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        # Ensure the file is opened in binary mode
        files["file"] = image_file
        data = {"ue_id": ue_id}
        print(send_post_request(get_inference_url(SERVER_URL), files, data))


def run_pressure_test(ue_id, num_requests):
    files = {}  # if files are required, use input to ask for the file path

    image_file_path = input("Please input the image file path: ")
    with open(image_file_path, "rb") as image_file:
        files["file"] = (
            image_file.read()
        )  # Read the file since we are sending multiple times.

    data = {"ue_id": ue_id}

    try:
        profile_result_processor = ProfileResultProcessor(SERVER_URL)

        for _ in range(num_requests):
            profile_response = send_post_request(get_profile_url(SERVER_URL), files, data)
            print(json.dumps(profile_response, indent=4))
            if not profile_response:
                print("No profile response received.")
                continue

            profile_result_processor.process_new_response(profile_response)

        # Print the final profile result
        profile_result_processor.print_profile_result()

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
            url = f"{SERVER_URL}/help"
            get_help(url)
        elif choice == "3":
            url = f"{SERVER_URL}/get_ue_log"
            ue_id = input("Enter the UE_ID: ")
            get_ue_log(url, ue_id)
        elif choice == "4":
            ue_id = input("Enter the UE_ID: ")
            num_requests = int(input("Enter the number of requests to send: "))
            run_pressure_test(ue_id, num_requests)
        elif choice == "5":
            print("Exiting the client. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
