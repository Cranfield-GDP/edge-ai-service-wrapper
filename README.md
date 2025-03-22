# AI WRAPPER

This repository contains all the tools to prepare the docker images for pre-trained AI models.
The docker images will contain a REST API server that serves the model along with other endpoints to fetch statistics.


## Steps

1. build and start the `edge_manager`. Check details in `edge_manager/README.md`.
2. build the AI model images, e.g., `docker build -t cranfield-edge-microsoft-resnet-50-api .`
3. start the AI model container, e.g., `docker run -d --name resnet50-api-container -p 8001:8000 cranfield-edge-microsoft-resnet-50-api`. Note that you must join the edge management network to be able to communicate with the edge manager.