from threading import Lock
import time


class Logger:
    """Logger class to handle logging for the AI model."""

    def __init__(self, container_id, container_name, model_name):
        self.container_id = container_id
        self.container_name = container_name
        self.model_name = model_name
        self.ue_logs = {}
        self.lock = Lock()

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
            return {
                "container_id": self.container_id,
                "container_name": self.container_name,
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
