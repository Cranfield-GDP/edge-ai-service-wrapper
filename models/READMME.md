To build the AI service images:
```bash
cd <a_model_folder>
docker build -t cranfield-edge-microsoft-resnet-50-api .
```

To start the AI service container:
```bash
docker run -d --name <name_of_the_container> -p 9000:8000 cranfield-edge-microsoft-resnet-50-api
``` 