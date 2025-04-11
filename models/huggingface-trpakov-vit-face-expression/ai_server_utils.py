import os
import socket
import torch
from io import BytesIO
import base64

from torch.profiler import profile, ProfilerActivity, record_function


# -------------------------------------------
# ENV Variables
# -------------------------------------------
NODE_ID = os.getenv("NODE_ID", socket.gethostname())
K8S_POD_NAME = os.getenv("K8S_POD_NAME", "UNKNOWN")


# -------------------------------------------
# Profile Utils
# -------------------------------------------
profile_activities = [
    ProfilerActivity.CPU,
    ProfilerActivity.CUDA,
    ProfilerActivity.MTIA,
    ProfilerActivity.XPU,
]

def process_model_output_logits(model, model_output_logits):
    """
    Process the model outputs to prepare for the response.
    """
    probabilities = torch.nn.functional.softmax(model_output_logits[0], dim=0)

    # Return the top 5 predictions with labels
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    predictions = []
    for i in range(top5_prob.size(0)):
        category_id = top5_catid[i].item()
        predictions.append(
            {
                "category_id": category_id,
                "label": model.config.id2label[category_id],
                "probability": top5_prob[i].item(),
            }
        )
    return predictions

def prepare_profile_results(prof):
    """
    Prepare the profile results for the model inputs and outputs.
    """
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    profile_event = prof.key_averages()[0]

    profile_result = {
        "name": profile_event.key,
        "device_type": str(profile_event.device_type),
        "device_name": str(profile_event.use_device),
        "cpu_memory_usage": profile_event.cpu_memory_usage,
        "self_cpu_memory_usage": profile_event.self_cpu_memory_usage,
        "device_memory_usage": profile_event.device_memory_usage,
        "self_device_memory_usage": profile_event.self_device_memory_usage,
        "cpu_time_total": profile_event.cpu_time_total,
        "self_cpu_time_total": profile_event.self_cpu_time_total,
        "device_time_total": profile_event.device_time_total,
        "self_device_time_total": profile_event.self_device_time_total,
    }
    return profile_result


def encode_image(image):
    """
    Encode the image to bytes
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_image


PROFILE_OUTPUT_JSON_SPEC = {
    "ue_id": "unique execution ID",
    "profile_result": {
        "name": "name of the profile event",
        "device_type": "type of device used (e.g., CPU, GPU, ...)",
        "device_name": "name of the device used",
        "cpu_memory_usage": "CPU memory usage in bytes",
        "self_cpu_memory_usage": "self CPU memory usage in bytes",
        "device_memory_usage": "device memory usage in bytes",
        "self_device_memory_usage": "self device memory usage in bytes",
        "cpu_time_total": "total CPU time in microseconds",
        "self_cpu_time_total": "self total CPU time in microseconds",
        "device_time_total": "total device time in microseconds",
        "self_device_time_total": "self total device time in microseconds",
    },
    "model_results": "the AI service model results",
}
