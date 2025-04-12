from pydantic import BaseModel
from typing import List, Optional


class InferenceProfile(BaseModel):
    cpu_time_ms: float
    device_time_ms: float
    cpu_memory_usage_MB: float
    self_cpu_memory_usage_MB: float
    device_memory_usage_MB: float
    self_device_memory_usage_MB: float
    energy_consumption_execution: float
    disk_IO_MB: float
    input_data_MB: float
    output_data_MB: float
    execution_time_ms: float
    execution_cost: float


class XAIProfile(BaseModel):
    xai_method: str
    cpu_time_ms: float
    device_time_ms: float
    cpu_memory_usage_MB: float
    self_cpu_memory_usage_MB: float
    device_memory_usage_MB: float
    self_device_memory_usage_MB: float
    energy_consumption_execution: float
    disk_IO_MB: float
    input_data_MB: float
    output_data_MB: float
    execution_time_ms: float
    execution_cost: float


class Profile(BaseModel):
    node_id: str
    device_type: str
    device_name: str
    initialization_time_ms: float
    eviction_time_ms: float
    initialization_cost: float
    keep_alive_cost: float
    energy_consumption_idle: float
    idle_container_cpu_memory_usage: str
    idle_container_device_memory_usage: str
    inference: InferenceProfile
    xai: Optional[List[XAIProfile]] = []


class Feedback(BaseModel):
    likes: List[str]
    dislikes: List[str]
    comments: List[str]


class Code(BaseModel):
    readme_content: str
    dockerfile_content: str
    ai_server_script_content: str
    ai_client_script_content: str
    ai_client_utils_script_content: Optional[str]
    model_script_content: Optional[str]
    healthcheck_script_content: Optional[str]


class AIService(BaseModel):
    model_name: str
    model_url: str
    task: str
    task_detail: str
    accuracy_info: str
    image_repository_url: str
    service_disk_size_bytes: float
    profiles: List[Profile]
    feedback: Feedback
    code: Code