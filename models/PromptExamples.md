### Prompt Exmaple to wrap hugging face models into docker image
```
I need to wrap the huggingface/microsoft/resnet-50 into a docker image, to be served by a RESTful API, e.g., using fastAPI server.

The REST API server will expose the following endpoints:
- POST /run to take necessary input (e.g., image or text or multi-model in a formdata) and a UE_ID (str) and and return a JSON response with the AI model's output. 
- GET /help to return a swagger-like JSON explaining the /run endpoint. this is to help the user to easily understand how to call the AI model

help me prepare the `ai_server.py`, `Dockerfile`, `ai_client.py` to build the AI service.

Below is an example for `ai_server.py`:
<replace this with the example code>

Below is an example for `Dockerfile`:
<replace this with the example code>

Below is an example for `ai_client.py`:
<replace this with the example code>
```