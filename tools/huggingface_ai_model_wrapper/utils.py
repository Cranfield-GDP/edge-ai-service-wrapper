import json
import os
import requests
import socket
from openai import OpenAI

AI_SERVER_SCRIPT_NAME = "ai_server.py"
AI_SERVER_UTILS_SCRIPT_NAME = "ai_server_utils.py"
AI_CLIENT_SCRIPT_NAME = "ai_client.py"
AI_CLIENT_UTILS_SCRIPT_NAME = "ai_client_utils.py"
MODEL_SCRIPT_NAME = "model.py"
XAI_MODEL_SCRIPT_NAME = "xai_model.py"
DOCKERFILE_NAME = "Dockerfile"
HEALTHCHECK_SCRIPT_NAME = "healthcheck.py"
SERVICE_DATA_JSON_NAME = "service_data.json"

NECESSARY_SERVICE_FILE_LIST = [
    AI_SERVER_SCRIPT_NAME,
    AI_CLIENT_SCRIPT_NAME,
    AI_CLIENT_UTILS_SCRIPT_NAME,
    AI_SERVER_UTILS_SCRIPT_NAME,
    DOCKERFILE_NAME,
    HEALTHCHECK_SCRIPT_NAME,
    MODEL_SCRIPT_NAME,
    SERVICE_DATA_JSON_NAME,
]

COMPLETE_SERVICE_FILE_LIST = NECESSARY_SERVICE_FILE_LIST + [
    XAI_MODEL_SCRIPT_NAME
]


def validate_hf_model_name(model_name: str) -> bool:
    """Validate the Hugging Face model name."""
    url = f"https://huggingface.co/api/models/{model_name}"
    response = requests.get(url)
    if response.status_code == 200:
        return True
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return False


def get_hf_model_directory(model_name: str) -> str:
    """Get the directory for the Hugging Face model."""
    # Assuming the model directory is structured as 'models/{model_name}'
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        f"huggingface-{model_name.replace("/", "-")}",
    )


def get_hf_model_readme(model_name: str) -> str:
    """Get the README file for the Hugging Face model."""
    url = f"https://huggingface.co/{model_name}/raw/main/README.md"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return ""


def copy_file_from_example_model_folder(
    file_to_copy: str,
    model_output_directory: str,
    example_content: dict,
    output_files_content: dict,
) -> None:
    """Copy the specified file from the example content."""
    file_path = os.path.join(model_output_directory, file_to_copy)
    with open(file_path, "w") as file:
        file.write(example_content[file_to_copy])
    output_files_content[file_to_copy] = example_content[file_to_copy]


def generate_model_script(
    model_name: str,
    model_readme,
    example_content: dict,
    output_files_content: dict,
    model_output_directory: str,
) -> None:
    """Generate the model.py file."""

    # --------------------------------
    # ask for additional help information from the user to be more flexible
    # --------------------------------
    additional_guidance = input(
        "Do you have any helpful guidance or instructions for the LLM agent to generate the model.py file?: "
    )
    if additional_guidance:
        additional_guidance = additional_guidance.strip()
    else:
        additional_guidance = ""
    
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""Help me generate the fastAPI router script for the fastAPI service which serves the pre-trained hugging face model {model_name}.
-----------------
Below is the README file of the hugging face model:
{model_readme}

-----------------
Below is an example model script content for your reference:
{example_content[MODEL_SCRIPT_NAME]}

{"-----------------\nBelow are additional guidance for you:\n" + additional_guidance if additional_guidance else ""}

-----------------
Requirements:
- follow the same endpoint design as the example model script.
- use the provided util functions from ai_server_utils as much as possible.
- if the AI model output binary content such as images, return the binary content in the response instead of saving it locally.
- Output only the raw content of the model script, without any additional text or explanation. 
- Do not wrap the output inside the ```python``` code block.""",
            }
        ],
        temperature=0.2,
    )

    model_script_content = completion.choices[0].message.content
    print(model_script_content)

    with open(os.path.join(model_output_directory, MODEL_SCRIPT_NAME), "w") as file:
        file.write(model_script_content)
    output_files_content[MODEL_SCRIPT_NAME] = model_script_content


def generate_dockerfile(
    model_name: str,
    model_readme,
    example_content: dict,
    output_files_content: dict,
    model_output_directory: str,
    additional_guidance: str = "",
) -> None:
    """Generate the Dockerfile."""

    # --------------------------------
    # ask for additional help information from the user to be more flexible
    # --------------------------------
    additional_guidance = input(
        "Do you have any helpful guidance or instructions for the LLM agent to generate the Dockerfile?: "
    )
    if additional_guidance:
        additional_guidance = additional_guidance.strip()
    else:
        additional_guidance = ""

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""Help me generate the Dockerfile to build the docker image containing fastAPI server script `ai_server.py`, serving the pre-trained hugging face model {model_name}.

-----------------
Below is the README file of the hugging face model:
{model_readme}

------------------
Below is the content of `ai_server.py`
{example_content[AI_SERVER_SCRIPT_NAME]}

-------------------
Below is an Dockerfile example for your reference:
{example_content[DOCKERFILE_NAME]}

{"-----------------\nBelow are additional guidance for you:\n" + additional_guidance if additional_guidance else ""}

--------------------
Requirements:
- follow the same design as the example Dockerfile.
- install the required dependencies for the API server and the hugging face model.
- output only the raw content of the Dockerfile, without any additional text or explanation.
- Do not wrap the output inside the ```dockerfile``` code block.""",
            }
        ],
        temperature=0.2,
    )

    dockerfile_content = completion.choices[0].message.content
    print(dockerfile_content)

    with open(os.path.join(model_output_directory, DOCKERFILE_NAME), "w") as file:
        file.write(dockerfile_content)
    output_files_content[DOCKERFILE_NAME] = dockerfile_content


def generate_ai_client_utils_script(
    model_name: str,
    model_readme,
    example_content: dict,
    output_files_content: dict,
    model_output_directory: str,
    additional_guidance: str = "",
) -> None:
    """Generate the AI client utils file."""

    # --------------------------------
    # ask for additional help information from the user to be more flexible
    # --------------------------------
    additional_guidance = input(
        "Do you have any helpful guidance or instructions for the LLM agent to generate the ai_client_utils.py file?: "
    )
    if additional_guidance:
        additional_guidance = additional_guidance.strip()
    else:
        additional_guidance = ""

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""Help me generate a util script for a FastAPI client to test an AI service ({model_name}) served by a fastAPI server.

-----------------
Below is the README file of the hugging face model:
{model_readme}

------------------
Below is the content of the `model.py` that serves the AI model:
{example_content[MODEL_SCRIPT_NAME]}

-------------------
Below is an example util script (for a image processing AI service) for your reference:
{example_content[AI_CLIENT_UTILS_SCRIPT_NAME]}

{"-----------------\nBelow are additional guidance for you:\n" + additional_guidance if additional_guidance else ""}

--------------------
Requirements:
- generate the two functions for preparing the two variables `files` and `data` respectively.
- these two functions will be called by the actual client script later.
- output only the raw content of the client util script, without any additional text or explanation.
- do not wrap the output inside the ```python``` code block.
""",
            }
        ],
    )

    ai_client_util_script_content = completion.choices[0].message.content
    print(ai_client_util_script_content)

    with open(
        os.path.join(model_output_directory, AI_CLIENT_UTILS_SCRIPT_NAME), "w"
    ) as file:
        file.write(ai_client_util_script_content)
    output_files_content[AI_CLIENT_UTILS_SCRIPT_NAME] = ai_client_util_script_content


def get_available_port() -> int:
    """Get an available port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def get_node_hostname() -> str:
    """Get the hostname of the node."""
    hostname = socket.gethostname()
    return hostname


def get_docker_image_build_name(model_name: str) -> str:
    """Get the docker image build name."""
    return f"cranfield-edge-{model_name.replace('/', '-')}:latest".lower()


def get_docker_container_run_name(model_name: str) -> str:
    """Get the docker container run name."""
    return f"cranfield-edge-{model_name.replace('/', '-')}-server".lower()


def download_model_readme(model_name: str) -> str:
    """Download the README file for the Hugging Face model."""
    readme_content = get_hf_model_readme(model_name)
    if readme_content:
        readme_file_path = os.path.join(get_hf_model_directory(model_name), "README.md")
        with open(readme_file_path, "w", encoding="utf-8") as file:
            file.write(readme_content)
        return readme_file_path
    else:
        raise Exception(f"Failed to download the README file for model {model_name}.")


def draft_model_task_detail(
    model_name: str,
    model_readme,
) -> str:
    """Let LLM to draft the task detail based on model's readme content, for better semantic searching."""
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""Help me write a detailed functionality description for pre-trained AI model ({model_name}).

-----------------
Below is the README file of the hugging face AI model:
{model_readme}

--------------------
Requirements:
- describe in detail the functionality of the AI model. For example, if the model is for image classification, you should describe the types of images it can classify, the expected input format, and the output format.
- write the description such that the content is suitable for embedding and semantic searching against potential use case scenarios. 
- output the content of the description only.
""",
            }
        ],
    )

    task_detail_content = completion.choices[0].message.content
    print(task_detail_content)
    return task_detail_content


def draft_model_accuracy_info(
    model_name: str,
    model_readme,
) -> str:
    """Let LLM to draft the accuracy info based on model's readme content."""
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""Help me extract any information about the (accuracy) performance for a pre-trained AI model ({model_name}) based on its README.

-----------------
Below is the README file of the hugging face AI model:
{model_readme}

--------------------
Requirements:
- if there are descriptions about the model's performance (accuracy), please describe them in natural language.
- if there are no such descriptions, just output "No accuracy information found."
- output the content of the description only.
""",
            }
        ],
    )

    acc_info_content = completion.choices[0].message.content
    print(acc_info_content)
    return acc_info_content


def prepare_service_data_json(model_name: str) -> str:
    """Copy a service_data.json for the model."""
    assert get_hf_model_directory(
        model_name
    ), f"Model directory not found for {model_name}."

    service_data_file = "service_data.json"
    output_path = os.path.join(get_hf_model_directory(model_name), service_data_file)

    source_path = os.path.join(
        os.path.dirname(__file__), "common_assets", service_data_file
    )
    assert os.path.exists(source_path), f"Template json file not found: {source_path}"

    with open(source_path, "rb") as source_file:
        service_data_json = json.load(source_file)

    service_data_json["model_name"] = model_name
    service_data_json["model_url"] = f"https://huggingface.co/{model_name}"
    service_data_json["task"] = get_model_pipeline_tag(model_name)

    model_readme = get_hf_model_readme(model_name)

    service_data_json["task_detail"] = draft_model_task_detail(
        model_name=model_name, model_readme=model_readme
    )
    service_data_json["accuracy_info"] = draft_model_accuracy_info(
        model_name=model_name, model_readme=model_readme
    )

    service_data_json["code"]["readme_content"] = model_readme
    service_data_json["code"]["dockerfile_content"] = open(
        os.path.join(get_hf_model_directory(model_name), DOCKERFILE_NAME), "r"
    ).read()
    service_data_json["code"]["ai_server_script_content"] = open(
        os.path.join(get_hf_model_directory(model_name), AI_SERVER_SCRIPT_NAME), "r"
    ).read()
    service_data_json["code"]["ai_server_utils_script_content"] = open(
        os.path.join(get_hf_model_directory(model_name), AI_SERVER_UTILS_SCRIPT_NAME),
        "r",
    ).read()
    service_data_json["code"]["ai_client_script_content"] = open(
        os.path.join(get_hf_model_directory(model_name), AI_CLIENT_SCRIPT_NAME), "r"
    ).read()
    service_data_json["code"]["ai_client_utils_script_content"] = open(
        os.path.join(get_hf_model_directory(model_name), AI_CLIENT_UTILS_SCRIPT_NAME),
        "r",
    ).read()
    service_data_json["code"]["model_script_content"] = open(
        os.path.join(get_hf_model_directory(model_name), MODEL_SCRIPT_NAME), "r"
    ).read()
    service_data_json["code"]["healthcheck_script_content"] = open(
        os.path.join(get_hf_model_directory(model_name), HEALTHCHECK_SCRIPT_NAME), "r"
    ).read()

    if os.path.exists(
        os.path.join(get_hf_model_directory(model_name), XAI_MODEL_SCRIPT_NAME)
    ):
        service_data_json["code"]["xai_model_script_content"] = open(
            os.path.join(get_hf_model_directory(model_name), XAI_MODEL_SCRIPT_NAME),
            "r",
        ).read()

    with open(output_path, "w") as dest_file:
        dest_file.write(json.dumps(service_data_json, indent=4))


def copy_test_image(model_name: str) -> str:
    """Copy a test image for the model."""
    assert get_hf_model_directory(
        model_name
    ), f"Model directory not found for {model_name}."

    # get all image files in the common_assets directory
    common_assets_dir = os.path.join(os.path.dirname(__file__), "common_assets")
    test_image_files = [
        file for file in os.listdir(common_assets_dir) if file.endswith(".png")
    ]
    assert len(test_image_files) > 0, f"No test image found in {common_assets_dir}."

    # ask the user to select a test image
    print("Please select a test image from the following list:")
    for i, file in enumerate(test_image_files):
        print(f"{i + 1}. {file}")
    selected_index = int(input("Enter the number of the selected image: ")) - 1
    assert (
        0 <= selected_index < len(test_image_files)
    ), f"Invalid selection: {selected_index + 1}. Please select a valid image number."
    test_image_file = test_image_files[selected_index]
    test_image_output_path = os.path.join(
        get_hf_model_directory(model_name), test_image_file
    )
    test_image_source_path = os.path.join(
        os.path.dirname(__file__), "common_assets", test_image_file
    )
    with open(test_image_source_path, "rb") as source_file:
        with open(test_image_output_path, "wb") as dest_file:
            dest_file.write(source_file.read())


def get_image_repository_name(model_name: str) -> str:
    """Get the name of the docker image."""
    ai_service_image_name = get_docker_image_build_name(model_name).replace(
        ":latest", ""
    )
    docker_username = os.getenv("DOCKER_USERNAME")
    image_repository_name = f"{docker_username}/{ai_service_image_name}"
    image_repository_name = image_repository_name.replace(" ", "_")
    return image_repository_name


def get_image_repository_full_url(model_name: str) -> str:
    """Get the full name of the docker image."""
    image_repository_name = get_image_repository_name(model_name)
    docker_registry = os.getenv("DOCKER_REGISTRY", "docker.io")
    docker_image_repository_url = f"{docker_registry}/{image_repository_name}"
    return docker_image_repository_url


def get_model_pipeline_tag(model_name: str) -> str:
    """Get the model pipeline tag."""
    url = f"https://huggingface.co/api/models/{model_name}"
    response = requests.get(url)
    if response.status_code == 200:
        model_info = response.json()
        return model_info.get("pipeline_tag", "unknown")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "unknown"


def validate_image_repository(model_name: str) -> bool:
    """Validate the image repository."""
    repository_name = get_image_repository_name(model_name)
    docker_hub_repo_api_url = (
        f"https://hub.docker.com/v2/repositories/{repository_name}"
    )
    response = requests.get(docker_hub_repo_api_url)
    if response.status_code == 200:
        image_data = response.json()
        if image_data.get("status_description") != "active":
            print(f"Repository {repository_name} is not active.")
            return False
        if image_data.get("repository_type") != "image":
            print(f"Repository {repository_name} is not an image repository.")
            return False

        return True
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return False


def update_ai_service_db(model_name: str) -> None:
    """Update the edge AI service database."""

    # validate the model name
    assert validate_hf_model_name(model_name), f"Invalid model name: {model_name}."
    print(f"Model name {model_name} is valid.")

    # validate the availability of docker image
    assert validate_image_repository(
        model_name
    ), f"Invalid image repository: {model_name}. Make sure you have successfully pushed the image to the repository."
    print(f"Image repository {model_name} is valid.")

    hf_model_directory = get_hf_model_directory(model_name)
    service_data_json_path = os.path.join(hf_model_directory, SERVICE_DATA_JSON_NAME)
    assert os.path.exists(
        service_data_json_path
    ), f"Service data json file not found: {service_data_json_path}"

    with open(service_data_json_path, "r") as file:
        service_data_json = json.load(file)

    # check if the AI service already exists in the database
    url = "http://localhost:8000/ai-services/"
    response = requests.get(url, params={"model_name": model_name})
    assert (
        response.status_code == 200
    ), f"Error fetching AI services: {response.status_code}, {response.text}"

    existing_services = response.json()
    assert (
        len(existing_services) <= 1
    ), f"Multiple services found for model {model_name}."
    if existing_services:
        existing_service = existing_services[0]

        # use put instead of post to update the existing service
        service_id = existing_service["id"]
        url = f"http://localhost:8000/ai-service/{service_id}"
        response = requests.put(url, json=service_data_json)
        if response.status_code == 200:
            print("AI service updated successfully.")
        else:
            print(f"Error updating AI service: {response.status_code}, {response.text}")
    else:
        # create a new service
        url = "http://localhost:8000/ai-services/"
        response = requests.post(url, json=service_data_json)
        if response.status_code == 201:
            print("AI service created successfully.")
        else:
            print(f"Error creating AI service: {response.status_code}, {response.text}")


def update_service_disk_size(model_name: str, size_in_bytes: int):
    service_data_json_file = os.path.join(
        get_hf_model_directory(model_name), "service_data.json"
    )
    with open(service_data_json_file, "r") as file:
        service_data = json.load(file)
        service_data["service_disk_size_bytes"] = size_in_bytes

    with open(service_data_json_file, "w") as file:
        json.dump(service_data, file, indent=4)
    print(f"Service disk size updated to {size_in_bytes} Bytes.")
