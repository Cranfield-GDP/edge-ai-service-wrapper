import base64
import json
import time
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from ai_client_utils import (
    prepare_ai_service_request_files,
    prepare_ai_service_request_data,
)

XAI_GRADCAM_METHODS = [
    "GradCAM",
    "HiResCAM",
    # "AblationCAM",
    "XGradCAM",
    "GradCAMPlusPlus",
    "ScoreCAM",
    "LayerCAM",
    "EigenCAM",
    "EigenGradCAM",
    "KPCA_CAM",
    "RandomCAM",
]

# -------------------------------------
# prompt for necessary inputs
# -------------------------------------
SERVER_URL = input("Please input server URL (default to http://localhost:9000): ")
if SERVER_URL.strip() == "":
    SERVER_URL = "http://localhost:9000"
UE_ID = input("Please input UE_ID (default to 123456): ")
if UE_ID.strip() == "":
    UE_ID = "123456"


def send_post_request(url, data, files):
    """Send request to run AI service and display AI service responses."""
    try:
        response = requests.post(url, files=files, data=data)
        # get the process time, node id and k8s pod name from the response headers
        process_time = response.headers.get("X-Process-Time")
        node_id = response.headers.get("X-NODE-ID")
        k8s_pod_name = response.headers.get("X-K8S-POD-NAME")
        if response.status_code == 200:
            return response.json(), process_time, node_id, k8s_pod_name
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def send_get_request(url, params=None):
    """Send GET request to the specified URL and return the response."""
    try:
        response = requests.get(url, params=params)
        process_time = response.headers.get("X-Process-Time")
        node_id = response.headers.get("X-NODE-ID")
        k8s_pod_name = response.headers.get("X-K8S-POD-NAME")
        if response.status_code == 200:
            return response.json(), process_time, node_id, k8s_pod_name
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


class ProfileResultProcessor:

    def __init__(self, server_url):
        self.server_url = server_url
        self.start_time = time.time()
        self.service_initialization_duration = 0
        self.response_counter = 0
        self.profile_name = None
        self.device_type = None
        self.device_name = None
        self.node_id = None
        self.k8s_pod_name = None
        self.cpu_memory_usage_bytes = 0
        self.self_cpu_memory_usage_bytes = 0
        self.device_memory_usage_bytes = 0
        self.self_device_memory_usage_bytes = 0
        self.cpu_time_total_us = 0
        self.self_cpu_time_total_us = 0
        self.device_time_total_us = 0
        self.self_device_time_total_us = 0

        # xai related
        self.gradcam_method_name = None

        self.fetch_service_initialization_duration()

    def fetch_service_initialization_duration(self):
        """Fetch the service initialization duration from the server."""
        response, process_time, node_id, k8s_pod_name = send_get_request(
            f"{self.server_url}/initialization_duration"
        )
        if response:
            self.service_initialization_duration = response.get(
                "initialization_duration", 0
            )
        else:
            print("Failed to fetch initialization duration.")
            self.service_initialization_duration = 0

    def process_new_response(
        self,
        profile_response,
        process_time=None,
        node_id=None,
        k8s_pod_name=None,
        gradcam_method_name=None,
    ):
        if not profile_response:
            return

        profile_result = profile_response["profile_result"]

        if not profile_result:
            return

        if self.profile_name is None:
            self.profile_name = profile_result["name"]
        if self.device_type is None:
            self.device_type = profile_result["device_type"]
        if self.device_name is None:
            self.device_name = profile_result["device_name"]
        if self.node_id is None:
            self.node_id = node_id
        if self.k8s_pod_name is None:
            self.k8s_pod_name = k8s_pod_name

        if self.gradcam_method_name is None:
            self.gradcam_method_name = gradcam_method_name

        # update the max profile result
        self.cpu_memory_usage_bytes = max(
            self.cpu_memory_usage_bytes, profile_result["cpu_memory_usage"]
        )
        # self cpu memory usage could be negative. here we take the value that has the max absolute value
        if abs(profile_result["self_cpu_memory_usage"]) > abs(
            self.self_cpu_memory_usage_bytes
        ):
            self.self_cpu_memory_usage_bytes = profile_result["self_cpu_memory_usage"]
        self.device_memory_usage_bytes = max(
            self.device_memory_usage_bytes, profile_result["device_memory_usage"]
        )
        # same as self cpu memory usage
        if abs(profile_result["self_device_memory_usage"]) > abs(
            self.self_device_memory_usage_bytes
        ):
            self.self_device_memory_usage_bytes = profile_result[
                "self_device_memory_usage"
            ]
        self.cpu_time_total_us = max(
            self.cpu_time_total_us, profile_result["cpu_time_total"]
        )
        self.self_cpu_time_total_us = max(
            self.self_cpu_time_total_us, profile_result["self_cpu_time_total"]
        )
        self.device_time_total_us = max(
            self.device_time_total_us, profile_result["device_time_total"]
        )
        self.self_device_time_total_us = max(
            self.self_device_time_total_us, profile_result["self_device_time_total"]
        )

        self.response_counter += 1

    def complete_profile(self):
        print("\n--------- PROFILE EVENT ---------\n")
        print(f"Name: {self.profile_name}")
        print(f"Device Type: {self.device_type}")
        print(f"Device Name: {self.device_name}")
        print(f"Node ID: {self.node_id}")
        print(f"K8S_POD_NAME: {self.k8s_pod_name}")

        if self.gradcam_method_name:
            print(f"GradCAM Method Name: {self.gradcam_method_name}")

        print("\n--------- LATENCY RESULT ---------\n")
        print("Service Initialization Duration: ", self.service_initialization_duration)
        print("Total Requests: ", self.response_counter)
        print(f"Total Time Taken: {time.time() - self.start_time:.2f} seconds")
        print(
            f"Average Time Taken: {(time.time() - self.start_time) / self.response_counter:.2f} seconds"
        )

        print("\n--------- RESOURCE USAGE ---------\n")
        print(f"CPU Memory Usage: {self.cpu_memory_usage_bytes / (1024 * 1024):.2f} MB")
        print(
            f"Self CPU Memory Usage: {self.self_cpu_memory_usage_bytes / (1024 * 1024):.2f} MB"
        )
        print(
            f"Device Memory Usage: {self.device_memory_usage_bytes / (1024 * 1024):.2f} MB"
        )
        print(
            f"Self Device Memory Usage: {self.self_device_memory_usage_bytes / (1024 * 1024):.2f} MB"
        )
        print(f"CPU Time Total: {self.cpu_time_total_us / 1000:.2f} ms")
        print(f"Self CPU Time Total: {self.self_cpu_time_total_us / 1000:.2f} ms")
        print(f"Device Time Total: {self.device_time_total_us / 1000:.2f} ms")
        print(f"Self Device Time Total: {self.self_device_time_total_us / 1000:.2f} ms")

        # update the service_data.json automatically
        with open("service_data.json", "r") as f:
            service_data = json.load(f)

        if not self.gradcam_method_name:
            complete_profile_data_to_save = {
                "node_id": self.node_id,
                "device_type": self.device_type,
                "device_name": self.device_name,
                "initialization_time_ms": self.service_initialization_duration * 1000,
                "eviction_time_ms": 0,
                "initialization_cost": 0,
                "keep_alive_cost": 0,
                "energy_consumption_idle": 0,
                "inference": {
                    "cpu_time_ms": self.cpu_time_total_us / 1000,
                    "device_time_ms": self.device_time_total_us / 1000,
                    "cpu_memory_usage_MB": self.cpu_memory_usage_bytes / (1024 * 1024),
                    "self_cpu_memory_usage_MB": self.self_cpu_memory_usage_bytes
                    / (1024 * 1024),
                    "device_memory_usage_MB": self.device_memory_usage_bytes
                    / (1024 * 1024),
                    "self_device_memory_usage_MB": self.self_device_memory_usage_bytes
                    / (1024 * 1024),
                    "energy_consumption_execution": 0,
                    "disk_IO_MB": 0,
                    "input_data_MB": 0,
                    "output_data_MB": 0,
                    "execution_time_ms": (time.time() - self.start_time)
                    / self.response_counter
                    * 1000,
                    "execution_cost": 0,
                },
            }

            # check if there is already a profile for this node id
            profile_found = False
            for profile in service_data["profiles"]:
                if profile["node_id"] == self.node_id:
                    profile["inference"] = complete_profile_data_to_save["inference"]
                    profile_found = True
                    break
            if not profile_found:
                service_data["profiles"].append(complete_profile_data_to_save)

            # save the updated service_data.json
            with open("service_data.json", "w") as f:
                json.dump(service_data, f, indent=4)
            print("\n--------- SERVICE DATA UPDATED ---------\n")

        else:
            complete_xai_profile_data_to_save = {
                "node_id": self.node_id,
                "device_type": self.device_type,
                "device_name": self.device_name,
                "initialization_time_ms": self.service_initialization_duration * 1000,
                "eviction_time_ms": 0,
                "initialization_cost": 0,
                "keep_alive_cost": 0,
                "energy_consumption_idle": 0,
                "xai": [
                    {
                        "xai_method": self.gradcam_method_name,
                        "cpu_time_ms": self.cpu_time_total_us / 1000,
                        "device_time_ms": self.device_time_total_us / 1000,
                        "cpu_memory_usage_MB": self.cpu_memory_usage_bytes
                        / (1024 * 1024),
                        "self_cpu_memory_usage_MB": self.self_cpu_memory_usage_bytes
                        / (1024 * 1024),
                        "device_memory_usage_MB": self.device_memory_usage_bytes
                        / (1024 * 1024),
                        "self_device_memory_usage_MB": self.self_device_memory_usage_bytes
                        / (1024 * 1024),
                        "energy_consumption_execution": 0,
                        "disk_IO_MB": 0,
                        "input_data_MB": 0,
                        "output_data_MB": 0,
                        "execution_time_ms": (time.time() - self.start_time)
                        / self.response_counter
                        * 1000,
                        "execution_cost": 0,
                    }
                ],
            }

            # check if there is already a profile for this node id
            profile_found = False
            for profile in service_data["profiles"]:
                if profile["node_id"] == self.node_id:
                    profile_found = True

                    # check if there is already a profile for this xai method
                    xai_method_found = False
                    if not profile.get("xai"):
                        profile["xai"] = []
                    for xai_profile in profile["xai"]:
                        if xai_profile["xai_method"] == self.gradcam_method_name:
                            xai_profile.update(
                                complete_xai_profile_data_to_save["xai"][0]
                            )
                            xai_method_found = True
                            break
                    if not xai_method_found:
                        profile["xai"].append(
                            complete_xai_profile_data_to_save["xai"][0]
                        )
                    break

            if not profile_found:
                service_data["profiles"].append(complete_xai_profile_data_to_save)

            # save the updated service_data.json
            with open("service_data.json", "w") as f:
                json.dump(service_data, f, indent=4)
            print("\n--------- SERVICE DATA UPDATED ---------\n")


def option_run():
    data = prepare_ai_service_request_data()
    files = prepare_ai_service_request_files()
    data = {**data, "ue_id": UE_ID}
    response, process_time, node_id, pod_name = send_post_request(
        f"{SERVER_URL}/model/run", data, files
    )
    print("Process Time: ", process_time)
    print("Node ID: ", node_id)
    print("K8S_POD_NAME: ", pod_name)
    print("Response")
    print(json.dumps(response, indent=4))

    # display the visualization image
    if response and "visualization" in response:
        visualization = response["visualization"]
        if visualization:
            image_bytes = base64.b64decode(visualization)
            image = Image.open(BytesIO(image_bytes))
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        else:
            print("No visualization image found in the response.")


def option_profile_run():
    data = prepare_ai_service_request_data()
    data = {**data, "ue_id": UE_ID}
    files = prepare_ai_service_request_files()
    num_requests = int(input("Enter the number of requests to send: "))

    try:
        profile_result_processor = ProfileResultProcessor(SERVER_URL)

        for _ in range(num_requests):
            profile_response, process_time, node_id, k8s_node_name = send_post_request(
                f"{SERVER_URL}/model/profile_run", data, files
            )
            print("Process Time: ", process_time)
            print("Node ID: ", node_id)
            print("K8S_POD_NAME: ", k8s_node_name)
            print("Response")
            print(json.dumps(profile_response, indent=4))
            if not profile_response:
                print("No profile response received.")
                continue

            profile_result_processor.process_new_response(
                profile_response,
                process_time=process_time,
                node_id=node_id,
                k8s_pod_name=k8s_node_name,
            )

        # Print the final profile result and update the service_data.json
        profile_result_processor.complete_profile()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def option_help():
    """Get help information from the server."""
    try:
        response, process_time, node_id, pod_name = send_get_request(
            f"{SERVER_URL}/help"
        )
        print("Process Time: ", process_time)
        print("Node ID: ", node_id)
        print("K8S_POD_NAME: ", pod_name)
        print("Response")
        print(json.dumps(response, indent=4))
    except Exception as e:
        print(f"Request failed: {e}")


def option_run_with_xai():
    """Run AI service with XAI."""
    data = prepare_ai_service_request_data()
    files = prepare_ai_service_request_files()

    print(
        "Note that currently only GradCAM methods on image-classification models are supported."
    )
    while True:
        gradcam_method_name = input(
            f"Please select a GradCAM method (options: {XAI_GRADCAM_METHODS}): "
        )
        if gradcam_method_name not in XAI_GRADCAM_METHODS:
            print(f"Invalid GradCAM method. Please select again.")
        else:
            break

    # ask for target class for explanation
    target_category_indexes = input(
        "Please input target category indexes for explanation (comma-separated, e.g., 111, 32, 44, ...): "
    )
    if not target_category_indexes or not target_category_indexes.strip():
        print(
            "No target category indexes provided. Defaulting to explaining the top confident category."
        )
        target_category_indexes = []
    else:
        target_category_indexes = [
            int(i.strip()) for i in target_category_indexes.split(",")
        ]

    data = {
        **data,
        "ue_id": UE_ID,
        "gradcam_method_name": gradcam_method_name,
        "target_category_indexes": target_category_indexes,
    }
    print("Data: ", data)
    response, process_time, node_id, k8s_pod_name = send_post_request(
        f"{SERVER_URL}/xai_model/run", data, files
    )
    print("Process Time: ", process_time)
    print("Node ID: ", node_id)
    print("K8S_POD_NAME: ", k8s_pod_name)

    # Handle JSON response
    model_results = response.get("model_results")
    if model_results:
        print("Model Results:", json.dumps(model_results, indent=4))

    xai_results = response.get("xai_results")
    if xai_results:
        print("XAI Results Method:", xai_results.get("xai_method"))

    # Handle binary image response
    encoded_image = xai_results.get("image")
    if encoded_image:
        image_bytes = base64.b64decode(encoded_image)

        # Load the image into Pillow
        image = Image.open(BytesIO(image_bytes))

        # Save image to disk
        image.save("xai_output.png")

        # Display the image using matplotlib
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def option_profile_run_with_xai():
    """Run AI service with XAI and profile the run."""
    data = prepare_ai_service_request_data()
    files = prepare_ai_service_request_files()
    print(
        "Note that currently only GradCAM methods on image-classification models are supported."
    )

    # ask for target class for explanation
    target_category_indexes = input(
        "Please input target category indexes for explanation (comma-separated, e.g., 111, 32, 44, ...): "
    )
    if not target_category_indexes or not target_category_indexes.strip():
        print(
            "No target category indexes provided. Defaulting to explaining the top confident category."
        )
        target_category_indexes = []
    else:
        target_category_indexes = [
            int(i.strip()) for i in target_category_indexes.split(",")
        ]

    num_requests = int(input("Enter the number of requests to send: "))

    try:
        for gradcam_method_name in XAI_GRADCAM_METHODS:

            data = {
                **data,
                "ue_id": UE_ID,
                "gradcam_method_name": gradcam_method_name,
                "target_category_indexes": target_category_indexes,
            }
            print("Data: ", data)

            profile_result_processor = ProfileResultProcessor(SERVER_URL)

            for _ in range(num_requests):
                response, process_time, node_id, k8s_pod_name = send_post_request(
                    f"{SERVER_URL}/xai_model/profile_run", data, files
                )
                print("Process Time: ", process_time)
                print("Node ID: ", node_id)
                print("K8S_POD_NAME: ", k8s_pod_name)
                if not response:
                    print("No profile response received.")
                    continue

                # Handle JSON response
                model_results = response.get("model_results")
                if model_results:
                    print("Model Results:", json.dumps(model_results, indent=4))

                profile_result_processor.process_new_response(
                    response,
                    process_time=process_time,
                    node_id=node_id,
                    k8s_pod_name=k8s_pod_name,
                    gradcam_method_name=gradcam_method_name,
                )

            # Print the final profile result and update the service_data.json
            profile_result_processor.complete_profile()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


OPTIONS = [
    {
        "label": "Get help information",
        "action": option_help,
    },
    {
        "label": "Run AI service",
        "action": option_run,
    },
    {
        "label": "Profile AI service",
        "action": option_profile_run,
    },
    {
        "label": "Run AI service with XAI (only image-classification models)",
        "action": option_run_with_xai,
    },
    {
        "label": "Profile AI service with XAI (only image-classification models)",
        "action": option_profile_run_with_xai,
    },
]


if __name__ == "__main__":
    while True:
        print("\nOptions:")
        for i, option in enumerate(OPTIONS, start=1):
            print(f"{i}. {option['label']}")
        print("q. Quit")
        choice = input("Enter your choice: ")

        if choice == "q":
            print("Exiting the client. Goodbye!")
            break
        else:
            try:
                choice = int(choice)
                if 1 <= choice <= len(OPTIONS):
                    OPTIONS[choice - 1]["action"]()
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
