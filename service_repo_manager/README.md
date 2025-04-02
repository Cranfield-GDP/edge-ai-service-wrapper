# Setup the edge manager server and the AI service database

```bash
docker compose build
docker compose up -d
```

# Edge AI Service Database
This is a Atlas MongoDB database that stores the AI services link and AI service metadata. The database is hosted locally.

# Edge AI Service Manager

This manager is built with FastAPI for serving RESTful API endpoints for interfacing with the AI service database. 

Current endpoints are:

- CRUD options for individual AI services

Check the swagger UI at `http://localhost:8000/docs` to see the available endpoints.


# utility script `service_manager_entrypoint.py`

* export all the AI services data into a local JSON file: <br>`python service_manager_entrypoint.py --option 1` 
