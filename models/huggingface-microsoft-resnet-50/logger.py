import os
import socket
from threading import Lock
import time

# -------------------------------------------
# Configuration
# -------------------------------------------
NODE_ID = os.getenv("NODE_ID", socket.gethostname())
K8S_POD_NAME = os.getenv("K8S_POD_NAME", "UNKNOWN")
print(f"Node ID: {NODE_ID}, K8S_POD_NAME: {K8S_POD_NAME}")


class Logger:
    """Singleton Logger class to handle logging for the AI model."""

    _instance = None  # Class-level attribute to hold the singleton instance
    _lock = Lock()  # Lock for thread-safe singleton initialization

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of Logger."""
        if not cls._instance:
            cls._instance = Logger()
        return cls._instance

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of Logger is created."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:  # Double-checked locking
                    cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name, node_id=NODE_ID, k8s_pod_name=K8S_POD_NAME):
        """Initialize the Logger instance."""
        # Avoid reinitializing if the instance already exists
        if not hasattr(self, "initialized"):
            self.model_name = model_name
            self.node_id = node_id
            self.k8s_pod_name = k8s_pod_name
            self.ue_logs = {}
            self.lock = Lock()
            self.initialized = True  # Mark the instance as initialized

    def add_ue_run_log(self, ue_id, input_size, execution_duration):
        """Add a run log entry for a specific UE_ID."""
        with self.lock:
            if ue_id not in self.ue_logs:
                self.ue_logs[ue_id] = {
                    "total_input_size": 0,
                    "total_execution_duration": 0.0,
                    "total_executions": 0,
                    "average_execution_duration": 0.0,
                    "latest_run": {},
                }
            self.ue_logs[ue_id]["total_input_size"] += input_size
            self.ue_logs[ue_id]["total_execution_duration"] += execution_duration
            self.ue_logs[ue_id]["total_executions"] += 1
            self.ue_logs[ue_id]["average_execution_duration"] = (
                self.ue_logs[ue_id]["total_execution_duration"]
                / self.ue_logs[ue_id]["total_executions"]
            )
            self.ue_logs[ue_id]["latest_run"] = {
                "input_size": input_size,
                "execution_duration": execution_duration,
                "timestamp": time.time(),
            }

    def get_ue_run_log(self, ue_id):
        """Retrieve the run log for a specific UE_ID."""
        with self.lock:
            if ue_id not in self.ue_logs:
                return None
            return {
                "node_id": self.node_id,
                "k8s_pod_name": self.k8s_pod_name,
                "model_name": self.model_name,
                "ue_id": ue_id,
                "total_input_size": self.ue_logs[ue_id]["total_input_size"],
                "total_execution_duration": self.ue_logs[ue_id][
                    "total_execution_duration"
                ],
                "total_executions": self.ue_logs[ue_id]["total_executions"],
                "average_execution_duration": self.ue_logs[ue_id][
                    "average_execution_duration"
                ],
                "latest_run": self.ue_logs[ue_id]["latest_run"],
            }