// This script will be mounted to the Mongo database's /docker-entrypoint-initdb.d directory
// and will be executed when the MongoDB container is started.
// Initialize MongoDB database (cranfield_ai_services) and collection (ai_services) 
// if they do not exist

// Load the MongoDB shell helper
const dbName = "cranfield_ai_services";
const collectionName = "ai_services";

// Switch to the target database
db = db.getSiblingDB(dbName);

// Check if the collection exists
if (!db.getCollectionNames().includes(collectionName)) {
    // Create the collection
    db.createCollection(collectionName);
    print(`Collection '${collectionName}' created in database '${dbName}'.`);
} else {
    print(`Collection '${collectionName}' already exists in database '${dbName}'.`);
}

print("Initialization script executed successfully.");