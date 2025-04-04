from pydantic import BaseModel
from typing import List, Optional, Dict


class ResourceProfile(BaseModel):
    cpu_time_ms: int
    cuda_time_ms: int
    xpu_time_ms: int
    mtia_time_ms: int
    ram_usage_MB: int
    vram_usage_MB: int
    energy_consumption_execution: int
    energy_consumption_idle: int
    disk_IO_MB: int
    input_data_MB: int
    output_data_MB: int
    service_disk_size: int


class LatencyProfile(BaseModel):
    initialization_time_ms: int
    inference_time_ms: int
    eviction_time_ms: int


class BillingProfile(BaseModel):
    initialization_cost: float
    keep_alive_cost: float
    execution_cost: float


class Profile(BaseModel):
    node_id: str
    resource: ResourceProfile
    latency: LatencyProfile
    billing: BillingProfile


class Feedback(BaseModel):
    likes: List[str]
    dislikes: List[str]
    comments: List[str]


class Code(BaseModel):
    readme_content: str
    dockerfile_content: str
    ai_server_script_content: str
    ai_client_script_content: str


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