from pydantic import BaseModel


class AIService(BaseModel):
    model_name: str
    model_url: str
    image_repository_url: str
    task: str
    readme_content: str
    ai_server_script_content: str
    ai_client_script_content: str
    dockerfile_content: str
    duration_per_run: float
    image_disk_size: float
    cpu_usage: float
    ram_usage: float
    gpu_usage: float = None
