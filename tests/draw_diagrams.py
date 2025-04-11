import json
import os
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------
# Common Data
# ---------------------------------------

COMMON_FILE_LIST = [
    "ai_client.py",
    "ai-server_utils.py",
    "ai_server.py",
    "healthcheck.py",
    "xai_model.py",
]

LLM_GENERATED_FILE_LIST = [
    "ai_client_utils.py",
    "model.py",
    "Dockerfile",
]

MODEL_NAMES = [
    "apple/mobilevit-small",
    "facebook/convnext-tiny-224",
    "facebook/regnet-y-040",
    "google/vit-base-patch16-224",
    "google/vit-large-patch32-384",
    "microsoft/cvt-13",
    "microsoft/resnet-50",
    "microsoft/swin-large-patch4-window12-384-in22k",
    "nvidia/mit-b0",
]

XAI_GRADCAM_METHODS = [
    "GradCAM",
    "HiResCAM",
    "XGradCAM",
    "GradCAMPlusPlus",
    "LayerCAM",
    "EigenCAM",
    "EigenGradCAM",
    "KPCA_CAM",
    "RandomCAM",
]


model_repository = os.path.join(os.path.dirname(__file__), "..", "models")

# ---------------------------------------
# Bar chart for lines of code comparison
# ---------------------------------------

# def get_file_lines_of_code(file_path):
#     """Get the number of lines in a file."""
#     try:
#         with open(file_path, 'r') as file:
#             return sum(1 for line in file)
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return 0


# sample_data = []

# for model_name in MODEL_NAMES:
#     model_path = os.path.join(model_repository, f"huggingface-{model_name.replace("/", "-")}")
#     if os.path.exists(model_path):
#         common_lines = sum([get_file_lines_of_code(os.path.join(model_path, file)) for file in COMMON_FILE_LIST if os.path.exists(os.path.join(model_path, file))])
#         llm_generated_lines = sum([get_file_lines_of_code(os.path.join(model_path, file)) for file in LLM_GENERATED_FILE_LIST if os.path.exists(os.path.join(model_path, file))])
#         manual_revision_lines = 0  # Placeholder for manual revision lines
#         sample_data.append({
#             "label": model_name,
#             "common_lines_of_code": common_lines,
#             "llm_generated_lines_of_code": llm_generated_lines,
#             "manual_revision_lines_of_code": manual_revision_lines
#         })

# print("Sample Data:")
# print(json.dumps(sample_data, indent=4))

# sample_data = [
#     {
#         "label": "microsoft/\nresnet-50",
#         "common_lines_of_code": 1077,
#         "llm_generated_lines_of_code": 147,
#         "manual_revision_lines_of_code": 0,
#     },
#     {
#         "label": "apple/\nmobilevit-small",
#         "common_lines_of_code": 1065,
#         "llm_generated_lines_of_code": 152,
#         "manual_revision_lines_of_code": 17,
#     },
#     {
#         "label": "facebook/\nconvnext-tiny-224",
#         "common_lines_of_code": 1069,
#         "llm_generated_lines_of_code": 146,
#         "manual_revision_lines_of_code": 21,
#     },
#     {
#         "label": "facebook/\nregnet-y-040",
#         "common_lines_of_code": 1064,
#         "llm_generated_lines_of_code": 148,
#         "manual_revision_lines_of_code": 8,
#     },
#     {
#         "label": "google/vit-\nbase-patch16-224",
#         "common_lines_of_code": 1070,
#         "llm_generated_lines_of_code": 149,
#         "manual_revision_lines_of_code": 25,
#     },
#     {
#         "label": "google/vit-\nlarge-patch32-384",
#         "common_lines_of_code": 1071,
#         "llm_generated_lines_of_code": 141,
#         "manual_revision_lines_of_code": 35,
#     },
#     {
#         "label": "microsoft/\ncvt-13",
#         "common_lines_of_code": 1080,
#         "llm_generated_lines_of_code": 149,
#         "manual_revision_lines_of_code": 22,
#     },
#     {
#         "label": "microsoft/swin-large\n-patch4-window12\n-384-in22k",
#         "common_lines_of_code": 1076,
#         "llm_generated_lines_of_code": 144,
#         "manual_revision_lines_of_code": 38,
#     },
#     {
#         "label": "nvidia/\nmit-b0",
#         "common_lines_of_code": 1078,
#         "llm_generated_lines_of_code": 145,
#         "manual_revision_lines_of_code": 30,
#     },
# ]


# # Extract data
# labels = [item["label"] for item in sample_data]
# common_lines = [item["common_lines_of_code"] for item in sample_data]
# llm_generated_lines = [item["llm_generated_lines_of_code"] for item in sample_data]
# manual_revision_lines = [item["manual_revision_lines_of_code"] for item in sample_data]

# # Bar width and positions
# bar_width = 0.25
# x_positions = np.arange(len(labels))

# # Plotting with smoother colors
# bars_common = plt.bar(
#     x_positions - bar_width,
#     common_lines,
#     width=bar_width,
#     label="Common Code",
#     color="#6baed6",
# )  # Light blue
# bars_llm = plt.bar(
#     x_positions,
#     llm_generated_lines,
#     width=bar_width,
#     label="LLM Generated Code",
#     color="#fd8d3c",
# )  # Soft orange
# bars_manual = plt.bar(
#     x_positions + bar_width,
#     manual_revision_lines,
#     width=bar_width,
#     label="Manually Revised Code",
#     color="#74c476",
# )  # Soft green

# # Add labels and title
# plt.xticks(x_positions, labels, rotation=45, ha="right")
# plt.ylabel("Lines of Code")
# plt.title("Lines of Code Comparison")
# plt.legend()

# # Add numbers to the bars
# for bars in [bars_common, bars_llm, bars_manual]:
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             height + 5,
#             str(int(height)),
#             ha="center",
#             va="bottom",
#         )

# # Show the plot
# plt.tight_layout()
# plt.show()

# ---------------------------------------
# Resource profiling
# ----------------------------------------

profile_machines = ["LAP004262", "ugurcan.celik"]

# resource_profiles = [
#     {
#         "model_name": "microsoft/resnet-50",
#         "profiles": {
#             "LAP004262": {
#                 "init_time": 5,
#                 "cpu_memory": 1000,
#                 "inference": {
#                     "cpu_time": 0.5,
#                     "device_time": 0,
#                     "avg_execution_time": 1
#                 },
#                 "xai": {
#                     "GradCAM": {
#                         "cpu_time": 1,
#                         "device_time": 0,
#                         "avg_execution_time": 2
#                     },
#                     "RandomCAM": {
#                         "cpu_time": 1,
#                         "device_time": 0,
#                         "avg_execution_time": 2
#                     },
#                     # ...
#                 }
#             },
#         }
#     }
# ]

resource_profiles = []

# collect the resource profile data
for model_name in MODEL_NAMES:
    model_path = os.path.join(
        model_repository, f"huggingface-{model_name.replace('/', '-')}"
    )
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."

    service_data_json_path = os.path.join(model_path, "service_data.json")
    assert os.path.exists(
        service_data_json_path
    ), f"Service data JSON path {service_data_json_path} does not exist."

    with open(service_data_json_path, "r") as f:
        service_data = json.load(f)

    model_profile = {
        "model_name": model_name,
        "profiles": {},
    }

    resource_profiles.append(model_profile)

    for machine_specific_profile in service_data["profiles"]:
        machine_name = machine_specific_profile["node_id"]
        if machine_name not in profile_machines:
            continue
        machine_profile = {
            "init_time": machine_specific_profile["initialization_time_ms"],
            "cpu_memory_gb": float(
                machine_specific_profile["idle_container_cpu_memory_usage"].replace(
                    "GB", ""
                )
            ),
            "inference": {
                "cpu_time": machine_specific_profile["inference"]["cpu_time_ms"],
                "device_time": machine_specific_profile["inference"]["device_time_ms"],
                "avg_execution_time": machine_specific_profile["inference"][
                    "execution_time_ms"
                ],
            },
            "xai": {},
        }

        for xai_profile in machine_specific_profile["xai"]:
            xai_name = xai_profile["xai_method"]
            xai_profile_data = {
                "cpu_time": xai_profile["cpu_time_ms"],
                "device_time": xai_profile["device_time_ms"],
                "avg_execution_time": xai_profile["execution_time_ms"],
            }
            machine_profile["xai"][xai_name] = xai_profile_data

        model_profile["profiles"][machine_name] = machine_profile

print("Resource Profiles:")
print(json.dumps(resource_profiles, indent=4))

# draw bar charts for initialization time, inference cpu time, execution time separately and a candlestick chart for the xai methods
# Replace model names with indices for x-axis labels
model_indices = list(range(len(resource_profiles)))
tick_labels = [i + 1 for i in model_indices]  # Create tick labels starting from 1

# Create a plot containing ten subplots
# the first row contains everything for the first machine, and the second row contains everything for the second machine
fig, axs = plt.subplots(2, 5, figsize=(20, 7))


for machine_index in range(2):

    # Initialization time
    for i, profile in enumerate(resource_profiles):
        init_time = (
            profile["profiles"][profile_machines[machine_index]]["init_time"] / 1000
        )  # Convert to seconds
        axs[machine_index, 0].bar(i, init_time, label=f"Model {i}")

    # add a subplot title below the x-axis
    axs[machine_index, 0].set_xlabel("(a) Initialization Time (s)" if machine_index == 0 else "(f) Initialization Time (s)")
    axs[machine_index, 0].set_xticks(model_indices)
    axs[machine_index, 0].set_xticklabels(
        tick_labels, rotation=0
    )  # Use indices as labels
    # axs[machine_index,  0].legend(loc="upper right")

    # Inference CPU time
    for i, profile in enumerate(resource_profiles):
        cpu_time = profile["profiles"][profile_machines[machine_index]]["inference"][
            "cpu_time"
        ]
        axs[machine_index, 1].bar(i, cpu_time, label=f"Model {i}")

    axs[machine_index, 1].set_xlabel("(b) Inference CPU Time (ms)" if machine_index == 0 else "(g) Inference CPU Time (ms)")
    axs[machine_index, 1].set_xticks(model_indices)
    axs[machine_index, 1].set_xticklabels(
        tick_labels, rotation=0
    )  # Use indices as labels
    # axs[machine_index,  1].legend(loc="upper right")

    # Inference execution time
    for i, profile in enumerate(resource_profiles):
        execution_time = (
            profile["profiles"][profile_machines[machine_index]]["inference"][
                "avg_execution_time"
            ]
            / 1000
        )  # Convert to seconds
        axs[machine_index, 2].bar(i, execution_time, label=f"Model {i}")

    axs[machine_index, 2].set_xlabel("(c) Request Response Time (s)" if machine_index == 0 else "(h) Request Response Time (s)")
    axs[machine_index, 2].set_xticks(model_indices)
    axs[machine_index, 2].set_xticklabels(
        tick_labels, rotation=0
    )  # Use indices as labels
    # axs[machine_index,  2].legend(loc="upper right")

    # Idel CPU Memory
    for i, profile in enumerate(resource_profiles):
        cpu_memory = profile["profiles"][profile_machines[machine_index]][
            "cpu_memory_gb"
        ]  # Convert to GB
        axs[machine_index, 4].bar(i, cpu_memory, label=f"Model {i}")
    axs[machine_index, 4].set_xlabel("(e) Idle CPU Memory (GB)" if machine_index == 0 else "(j) Idle CPU Memory (GB)")
    axs[machine_index, 4].set_xticks(model_indices)
    axs[machine_index, 4].set_xticklabels(
        tick_labels, rotation=0
    )  # Use indices as labels

    # XAI average execution time
    for i, profile in enumerate(resource_profiles):
        xai_methods = profile["profiles"][profile_machines[machine_index]]["xai"]
        xai_times = [
            xai_methods[xai_name]["avg_execution_time"] / 1000
            for xai_name in xai_methods
        ]
        # Calculate min, max, mean, and median for candlestick
        xai_min = np.min(xai_times)
        xai_max = np.max(xai_times)
        xai_mean = np.mean(xai_times)
        xai_median = np.median(xai_times)

        # Draw candlestick: vertical line for min to max, box for mean to median
        axs[machine_index, 3].plot(
            [i, i], [xai_min, xai_max], color="black", linewidth=1
        )  # Vertical line
        axs[machine_index, 3].bar(
            i,
            xai_median - xai_mean,
            bottom=xai_mean,
            width=0.8,
            color="blue",
            alpha=0.6,
        )  # Box for mean to median

    axs[machine_index, 3].set_xlabel("(d) XAI Response Time (s)" if machine_index == 0 else "(i) XAI Response Time (s)")
    axs[machine_index, 3].set_xticks(model_indices)
    axs[machine_index, 3].set_xticklabels(
        tick_labels, rotation=0
    )  # Use indices as labels
    # axs[3].legend(loc="upper right")

# Adjust layout
# plt.tight_layout()
plt.show()
