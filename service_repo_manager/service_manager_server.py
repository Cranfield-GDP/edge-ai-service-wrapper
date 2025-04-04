from typing import Optional
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId
from api_models import AIService
import os

# Initialize FastAPI app with metadata for Swagger UI
app = FastAPI(
    title="AI Service Manager API",
    description="API for managing AI services in the Manager system. This includes creating, reading, updating, and deleting AI service stored in MongoDB.",
    version="0.0.1",
    contact={
        "name": "AI Service Manager",
        "email": "yun.tang@cranfield.ac.uk",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/?directConnection=true")
client = MongoClient(MONGO_URI)
db = client.get_database("cranfield_ai_services")
collection = db["ai_services"]

# Helper function to serialize MongoDB documents
def serialize_ai_service(doc):
    doc = {**doc, "id": str(doc["_id"])}
    doc.pop("_id", None)  # Remove the MongoDB ObjectId field
    return doc

# Create AI Service Record
@app.post("/ai-services/", response_model=dict, status_code=201, tags=["AI Service"])
async def create_ai_service(ai_service: AIService):
    """
    Create a new AI Service.
    """
    result = collection.insert_one(ai_service.model_dump())
    created = collection.find_one({"_id": result.inserted_id})
    return serialize_ai_service(created)

# Read all AI Service with optional filtering
@app.get("/ai-services/", response_model=list, status_code=200, tags=["AI Service"])
async def get_all_ai_services(model_name: Optional[str] = None, task: Optional[str] = None):
    """
    Retrieve all AI Service with optional filtering.
    """
    query = {}
    if model_name:
        query["model_name"] = model_name
    if task:
        query["task"] = task

    services = collection.find(query)
    return [serialize_ai_service(s) for s in services]

# Read a single AI Service by ID
@app.get("/ai-service/{service_id}", response_model=dict, status_code=200, tags=["AI Service"])
async def get_ai_service(service_id: str):
    """
    Retrieve a single AI Service by its ID.

    - **service_id**: The ID of the AI service.
    """
    service = collection.find_one({"_id": ObjectId(service_id)})
    if not service:
        raise HTTPException(status_code=404, detail="AI Service not found")
    return serialize_ai_service(service)

# Update AI Service by ID
@app.put("/ai-service/{service_id}", response_model=dict, status_code=200, tags=["AI Service"])
async def update_ai_service(service_id: str, ai_service: AIService):
    """
    Update an existing AI Service by its ID.

    - **service_id**: The ID of the AI service.
    - **ai_service**: The updated AI service data.
    """
    result = collection.update_one(
        {"_id": ObjectId(service_id)}, {"$set": ai_service.model_dump()}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="AI Service not found")
    updated = collection.find_one({"_id": ObjectId(service_id)})
    return serialize_ai_service(updated)

# Delete AI Service by ID
@app.delete("/ai-service/{service_id}", response_model=dict, status_code=200, tags=["AI Service"])
async def delete_ai_service(service_id: str):
    """
    Delete an AI Service by its ID.
    - **service_id**: The ID of the AI service.
    """
    result = collection.delete_one({"_id": ObjectId(service_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="AI Service not found")
    return {"message": "AI Service deleted successfully"}