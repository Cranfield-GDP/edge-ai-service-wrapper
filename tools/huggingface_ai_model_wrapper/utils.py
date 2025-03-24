import os
import requests

AI_SERVER_SCRIPT_NAME = "ai_server.py"
AI_CLIENT_SCRIPT_NAME = "ai_client.py"
DOCKERFILE_NAME = "Dockerfile"
LOGGER_SCRIPT_NAME = "logger.py"

TARGET_FILES_TO_GENERATE = [
    AI_SERVER_SCRIPT_NAME,
    AI_CLIENT_SCRIPT_NAME,
    DOCKERFILE_NAME,
    LOGGER_SCRIPT_NAME,
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


def copy_logger_file(
    model_output_directory: str, example_content: dict, output_files_content: dict
) -> None:
    """Copy the logger.py file directly."""
    logger_file_path = os.path.join(model_output_directory, LOGGER_SCRIPT_NAME)
    with open(logger_file_path, "w") as file:
        file.write(example_content[LOGGER_SCRIPT_NAME])
    output_files_content[LOGGER_SCRIPT_NAME] = example_content[LOGGER_SCRIPT_NAME]


def generate_ai_server_script(
    model_name: str,
    model_readme,
    example_content: dict,
    output_files_content: dict,
    model_output_directory: str,
) -> None:
    """Generate the AI server file."""
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""Help me generate the fastAPI server script, serving the pre-trained hugging face model {model_name}.
-----------------                
Below is the README file of the hugging face model:
{model_readme}

-----------------
Below is an example server script content for your reference:
{example_content[AI_SERVER_SCRIPT_NAME]}

-----------------
Requirements:
- follow the same endpoint design as the example server script.
- log each request in the same way as the example server script.
- if the AI model output binary content such as images, return the binary content in the response instead of saving it locally.
- Output only the raw content of the server script, without any additional text or explanation. 
- Do not wrap the output inside the ```python``` code block.""",
            }
        ],
        temperature=0.2,
    )

    ai_server_content = completion.choices[0].message.content
    print(ai_server_content)

    with open(os.path.join(model_output_directory, AI_SERVER_SCRIPT_NAME), "w") as file:
        file.write(ai_server_content)
    output_files_content[AI_SERVER_SCRIPT_NAME] = ai_server_content


def generate_dockerfile(
    model_name: str,
    model_readme,
    example_content: dict,
    output_files_content: dict,
    model_output_directory: str,
) -> None:
    """Generate the Dockerfile."""
    from openai import OpenAI

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


def generate_ai_client_script(
    model_name: str,
    model_readme,
    example_content: dict,
    output_files_content: dict,
    model_output_directory: str,
) -> None:
    """Generate the AI client file."""
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""Help me generate a client script to test an AI service ({model_name}) served by a fastAPI server.

-----------------
Below is the README file of the hugging face model:
{model_readme}

------------------
Below is the content of the `ai_server.py`
{example_content[AI_SERVER_SCRIPT_NAME]}

-------------------
Below is an example client script (for a image processing AI service) for your reference:
{example_content[AI_CLIENT_SCRIPT_NAME]}

--------------------
Requirements:
- follow similar commandl line interface design as the example client script.
- test all the endpoints defined in the server script.
- If the AI model outputs binary content such as images, save the binary content to a file and print the save path.
- Do not run AI model in the client script.
- output only the raw content of the client script, without any additional text or explanation.
- do not wrap the output inside the ```python``` code block.
""",
            }
        ],
    )

    ai_client_content = completion.choices[0].message.content
    print(ai_client_content)

    with open(os.path.join(model_output_directory, AI_CLIENT_SCRIPT_NAME), "w") as file:
        file.write(ai_client_content)
    output_files_content[AI_CLIENT_SCRIPT_NAME] = ai_client_content


def get_available_port() -> int:
    """Get an available port."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


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
        with open(readme_file_path, "w") as file:
            file.write(readme_content)
        return readme_file_path
    else:
        raise Exception(f"Failed to download the README file for model {model_name}.")


def copy_test_image(model_name: str) -> str:
    """Copy a test image for the model."""
    assert get_hf_model_directory(
        model_name
    ), f"Model directory not found for {model_name}."

    test_image_file = "puppy.png"
    test_image_output_path = os.path.join(
        get_hf_model_directory(model_name), test_image_file
    )

    test_image_source_path = os.path.join(
        os.path.dirname(__file__), "test_assets", test_image_file
    )
    assert os.path.exists(
        test_image_source_path
    ), f"Test image not found: {test_image_source_path}"

    with open(test_image_source_path, "rb") as source_file:
        with open(test_image_output_path, "wb") as dest_file:
            dest_file.write(source_file.read())


def get_image_repository_name(model_name: str) -> str:
    """Get the name of the docker image."""
    ai_service_image_name = get_docker_image_build_name(model_name).replace(":latest", "")
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
    docker_hub_repo_api_url = f"https://hub.docker.com/v2/repositories/{repository_name}"
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


def update_edge_ai_service_db(model_name: str) -> None:
    """Update the edge AI service database."""

    # validate the model name
    assert validate_hf_model_name(model_name), f"Invalid model name: {model_name}."
    print(f"Model name {model_name} is valid.")

    # validate the availability of docker image
    assert validate_image_repository(
        model_name
    ), f"Invalid image repository: {model_name}. Make sure you have successfully pushed the image to the repository."
    print(f"Image repository {model_name} is valid.")

    duration_per_run = input("Enter the duration per run (in seconds): ")
    cpu_usage = input("Enter the CPU usage under pressure testing (in percentage): ")
    ram_usage = input("Enter the RAM usage under pressure testing (in MB): ")
    gpu_usage = input("Enter the GPU usage under pressure testing (in MB): ")
    image_disk_size = input("Enter the image disk size (in GB): ")

    ai_service_image_data = {
        "model_name": model_name,
        "model_url": f"https://huggingface.co/{model_name}",
        "image_repository_url": get_image_repository_full_url(model_name),
        "task": get_model_pipeline_tag(model_name),
        "readme_content": get_hf_model_readme(model_name),
        "ai_server_script_content": open(
            os.path.join(get_hf_model_directory(model_name), AI_SERVER_SCRIPT_NAME), "r"
        ).read(),
        "ai_client_script_content": open(
            os.path.join(get_hf_model_directory(model_name), AI_CLIENT_SCRIPT_NAME), "r"
        ).read(),
        "dockerfile_content": open(
            os.path.join(get_hf_model_directory(model_name), DOCKERFILE_NAME), "r"
        ).read(),
        "duration_per_run": duration_per_run,
        "image_disk_size": image_disk_size,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "gpu_usage": gpu_usage,
    }

    # check if the AI service image already exists in the database
    url = "http://localhost:8000/ai-service-images/"
    response = requests.get(url, params={"model_name": model_name})
    assert response.status_code == 200, f"Error fetching AI service images: {response.status_code}, {response.text}"

    existing_images = response.json()
    assert len(existing_images) <= 1, f"Multiple images found for model {model_name}."
    if existing_images:
        existing_image = existing_images[0]

        # use put instead of post to update the existing image
        image_id = existing_image["id"]
        url = f"http://localhost:8000/ai-service-images/{image_id}"
        response = requests.put(url, json=ai_service_image_data)
        if response.status_code == 200:
            print("AI service image updated successfully.")
        else:
            print(f"Error updating AI service image: {response.status_code}, {response.text}")
    else:
        # create a new image
        url = "http://localhost:8000/ai-service-images/"
        response = requests.post(url, json=ai_service_image_data)
        if response.status_code == 201:
            print("AI service image created successfully.")
        else:
            print(f"Error creating AI service image: {response.status_code}, {response.text}")