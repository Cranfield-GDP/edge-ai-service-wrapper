import json
import time
import requests

def get_help(url):
    """Get help information from the server."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Help Information:")
            print(response.json())
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def get_ue_log(url, ue_id):
    """Retrieve logs for a specific UE_ID."""
    # url = f"{SERVER_URL}/get_ue_log"
    try:
        response = requests.get(url, params={"ue_id": ue_id})
        if response.status_code == 200:
            print(f"Logs for UE_ID {ue_id}:")
            print(response.json())
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def get_inference_url(server_url):
    return f"{server_url}/model/run"

def get_profile_url(server_url):
    return f"{server_url}/model/profile"

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

        self.update_service_initialization_duration()
    
    def update_service_initialization_duration(self):
        """Fetch the service initialization duration from the server."""
        try:
            response = requests.get(f"{self.server_url}/initialization_duration")
            if response.status_code == 200:
                self.service_initialization_duration = response.json().get("initialization_duration", 0)
            else:
                print(f"Error fetching initialization duration: {response.status_code}, {response.text}")
                return 0
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return 0

    def process_new_response(self, profile_response):
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
            self.node_id = profile_response["node_id"]
        if self.k8s_pod_name is None:
            self.k8s_pod_name = profile_response["k8s_pod_name"]

        # update the max profile result
        self.cpu_memory_usage_bytes = max(
            self.cpu_memory_usage_bytes, profile_result["cpu_memory_usage"]
        )
        self.self_cpu_memory_usage_bytes = max(
            self.self_cpu_memory_usage_bytes, profile_result["self_cpu_memory_usage"]
        )
        self.device_memory_usage_bytes = max(
            self.device_memory_usage_bytes, profile_result["device_memory_usage"]
        )
        self.self_device_memory_usage_bytes = max(      
            self.self_device_memory_usage_bytes, profile_result["self_device_memory_usage"]
        )
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
    
    def print_profile_result(self):
        print("\n--------- PROFILE EVENT ---------\n")
        print(f"Name: {self.profile_name}")
        print(f"Device Type: {self.device_type}")
        print(f"Device Name: {self.device_name}")
        print(f"Node ID: {self.node_id}")
        print(f"K8S_POD_NAME: {self.k8s_pod_name}")    
        
        print("\n--------- LATENCY RESULT ---------\n")
        print("Service Initialization Duration: ", self.service_initialization_duration)
        print("Total Requests: ", self.response_counter)
        print(
            f"Total Time Taken: {time.time() - self.start_time:.2f} seconds"
        )
        print(
            f"Average Time Taken: {(time.time() - self.start_time) / self.response_counter:.2f} seconds"
        )

        print("\n--------- RESOURCE USAGE ---------\n")
        print(
            f"CPU Memory Usage: {self.cpu_memory_usage_bytes / (1024 * 1024):.2f} MB"
        )
        print(
            f"Self CPU Memory Usage: {self.self_cpu_memory_usage_bytes / (1024 * 1024):.2f} MB"
        )
        print(
            f"Device Memory Usage: {self.device_memory_usage_bytes / (1024 * 1024):.2f} MB"
        )
        print(
            f"Self Device Memory Usage: {self.self_device_memory_usage_bytes / (1024 * 1024):.2f} MB"
        )
        print(
            f"CPU Time Total: {self.cpu_time_total_us / 1000:.2f} ms"
        )
        print(
            f"Self CPU Time Total: {self.self_cpu_time_total_us / 1000:.2f} ms"
        )
        print(
            f"Device Time Total: {self.device_time_total_us / 1000:.2f} ms"
        )
        print(
            f"Self Device Time Total: {self.self_device_time_total_us / 1000:.2f} ms"
        )

        # update the service_data.json automatically
        with open("service_data.json", "r") as f:
            service_data = json.load(f)
        
        profile_data_to_save = {
            "node_id": self.node_id,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "resource": {
                "cpu_time_ms": self.cpu_time_total_us / 1000,
                "device_time_ms": self.device_time_total_us / 1000,
                "cpu_memory_usage_MB": max(self.cpu_memory_usage_bytes, self.self_cpu_memory_usage_bytes) / (1024 * 1024),
                "device_memory_usage_MB": max(self.device_memory_usage_bytes, self.self_device_memory_usage_bytes) / (1024 * 1024),
                "energy_consumption_execution": 0,
                "energy_consumption_idle": 0,
                "disk_IO_MB": 0,
                "input_data_MB": 0,
                "output_data_MB": 0
            },
            "latency": {
                "initialization_time_ms": self.service_initialization_duration * 1000,
                "inference_time_ms": (time.time() - self.start_time) / self.response_counter * 1000,
                "eviction_time_ms": 0
            },
            "billing": {
                "initialization_cost": 0,
                "keep_alive_cost": 0,
                "execution_cost": 0
            }
        }

        # check if there is already a profile for this node id
        for profile in service_data["profiles"]:
            if profile["node_id"] == self.node_id:
                # update the profile
                profile.update(profile_data_to_save)
                break
        else:
            # add a new profile
            service_data["profiles"].append(profile_data_to_save)
        
        # save the updated service_data.json
        with open("service_data.json", "w") as f:
            json.dump(service_data, f, indent=4)
        print("\n--------- SERVICE DATA UPDATED ---------\n")   
        
        


