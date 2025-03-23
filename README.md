# AI WRAPPER

This repository contains all the tools to prepare the docker images for pre-trained AI models.

AI Service Docker Images:
--- 
The docker images will contain a REST API server that serves the model along with other endpoints to fetch statistics.

All the AI services are stored under `models/` folder.

Edge AI Service Manager:
---
This is a FastAPI-based server for managing AI services data, e.g., CRUD endpoints for invidiual AI service. Check out `edge_manager/README.md` for more details.

Tools (under `tools/` folder):
---
- Helper scripts and GUI to prepare the docker images for pre-trained AI models on HuggingFace Hub. Checkout `tools/huggingface_ai_model_wrapper/entrypoint_ui.py` for more details.


