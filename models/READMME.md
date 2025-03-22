To build the AI model images:
```bash
cd a_model_folder
docker build -t cranfield-edge-microsoft-resnet-50-api .
```

To start the AI model container:
```bash
docker run -d --name resnet50-api-container -p 8001:8000 cranfield-edge-microsoft-resnet-50-api
``` 