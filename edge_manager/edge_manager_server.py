from typing import Optional
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId
from api_models import AIServiceImage
import os

# Initialize FastAPI app with metadata for Swagger UI
app = FastAPI(
    title="Edge Manager API",
    description="API for managing AI service images in the Edge Manager system. This includes creating, reading, updating, and deleting AI service images stored in MongoDB.",
    version="0.0.1",
    contact={
        "name": "Edge Manager",
        "email": "yun.tang@cranfield.ac.uk",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/ai_services_db")
client = MongoClient(MONGO_URI)
db = client.get_database()
collection = db["ai_service_images"]

# Helper function to serialize MongoDB documents
def serialize_ai_service_image(doc):
    doc = {**doc, "id": str(doc["_id"])}
    doc.pop("_id", None)  # Remove the MongoDB ObjectId field
    return doc

# Create AI Service Image
@app.post("/ai-service-images/", response_model=dict, status_code=201, tags=["AI Service Images"])
async def create_ai_service_image(ai_service_image: AIServiceImage):
    """
    Create a new AI Service Image.

    - **model_name**: Name of the AI model.
    - **task**: Task performed by the AI model (e.g., image classification, object detection).
    - **docker_image**: Docker image name for the AI service.
    """
    result = collection.insert_one(ai_service_image.model_dump())
    created = collection.find_one({"_id": result.inserted_id})
    return serialize_ai_service_image(created)

# Read all AI Service Images with optional filtering
@app.get("/ai-service-images/", response_model=list, status_code=200, tags=["AI Service Images"])
async def get_all_ai_service_images(model_name: Optional[str] = None, task: Optional[str] = None):
    """
    Retrieve all AI Service Images with optional filtering.

    - **model_name**: Filter by model name.
    - **task**: Filter by task type.
    """
    query = {}
    if model_name:
        query["model_name"] = model_name
    if task:
        query["task"] = task

    images = collection.find(query)
    return [serialize_ai_service_image(image) for image in images]

# Read a single AI Service Image by ID
@app.get("/ai-service-images/{image_id}", response_model=dict, status_code=200, tags=["AI Service Images"])
async def get_ai_service_image(image_id: str):
    """
    Retrieve a single AI Service Image by its ID.

    - **image_id**: The ID of the AI service image.
    """
    image = collection.find_one({"_id": ObjectId(image_id)})
    if not image:
        raise HTTPException(status_code=404, detail="AI Service Image not found")
    return serialize_ai_service_image(image)

# Update AI Service Image by ID
@app.put("/ai-service-images/{image_id}", response_model=dict, status_code=200, tags=["AI Service Images"])
async def update_ai_service_image(image_id: str, ai_service_image: AIServiceImage):
    """
    Update an existing AI Service Image by its ID.

    - **image_id**: The ID of the AI service image.
    - **ai_service_image**: The updated AI service image data.
    """
    result = collection.update_one(
        {"_id": ObjectId(image_id)}, {"$set": ai_service_image.model_dump()}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="AI Service Image not found")
    updated = collection.find_one({"_id": ObjectId(image_id)})
    return serialize_ai_service_image(updated)

# Delete AI Service Image by ID
@app.delete("/ai-service-images/{image_id}", response_model=dict, status_code=200, tags=["AI Service Images"])
async def delete_ai_service_image(image_id: str):
    """
    Delete an AI Service Image by its ID.

    - **image_id**: The ID of the AI service image.
    """
    result = collection.delete_one({"_id": ObjectId(image_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="AI Service Image not found")
    return {"message": "AI Service Image deleted successfully"}