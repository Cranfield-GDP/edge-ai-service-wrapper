### Prompt Exmaple to wrap hugging face models into docker image
```
I need to wrap the huggingface/microsoft/resnet-50 into a docker image, to be served by a RESTful API, e.g., using fastAPI server. 

the REST API server will expose the following endpoints:
- POST /run to take necessary input (e.g., image or text or multi-model in a formdata) and and return a JSON response with the AI model's output. 
- GET /help to return a swagger-like JSON explaining the /run endpoint. this is to help the user to easily understand how to call the AI model
- GET /resource to return a JSON containing the docker container's occupied CPU, RAM and GPU resource.

help me prepare the `server.py`, `Dockerfile` to build the docker image. 

requirements for `server.py`:
- The AI server needs to log each run containing, e.g., id of the container, name of the AI model, type and size of input (excluding the raw input content), type and size of output, execution time, and amount of CPU/GPU resource used (if available) by wrapping the data inside a JSON and post it to a log server running within the same edge cluster of a given IP,  e.g., `{EDGE_LOGGER_SERVER_IP}:{EDGE_LOGGER_SERVER_PORT}/log'
- get the edge logger server IP and port from the environment variable `EDGE_LOGGER_SERVER_IP` and `EDGE_LOGGER_SERVER_PORT` respectively.

requirements for `Dockerfile`:
- Install all the dependencies inside Dockerfile, instead of using a separate `requirements.txt` file. 
- The size of the built docker image image should be as small as possible. 
- The docker image should be based on the latest version of the official python image, e.g., `python:3.12-slim`.

```