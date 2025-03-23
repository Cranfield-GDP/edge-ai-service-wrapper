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
Below is an example client script content for your reference:
{example_content[AI_CLIENT_SCRIPT_NAME]}

--------------------
Requirements:
- follow the same design as the example client script.
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