import argparse
import os
import json
import time
from pymongo import MongoClient


DEFAULT_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "database_storage", f"cranfield_ai_services__{time.time()}.json")
DEFAULT_MONGO_URI = "mongodb://user:pass@localhost:27017/?directConnection=true"

def export_mongodb_to_json(uri, database_name, collection_name, output_file):
    """
    Connects to MongoDB and exports data from the specified collection to a JSON file.
    """
    try:
        print(f"Connecting to MongoDB at {uri}...")
        client = MongoClient(uri)
        db = client[database_name]
        collection = db[collection_name]

        print(f"Exporting data from {database_name}/{collection_name}...")
        data = list(collection.find())
        # Remove MongoDB's internal '_id' field for JSON compatibility
        for record in data:
            record.pop('_id', None)

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        print(f"Data successfully exported to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

def main():
    parser = argparse.ArgumentParser(description="Edge Manager Entrypoint Script")
    parser.add_argument(
        "--option",
        type=int,
        required=True,
        help="Choose an option: 1 - Export MongoDB data to JSON"
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=DEFAULT_MONGO_URI,
        help=f"MongoDB connection URI (default: {DEFAULT_MONGO_URI})"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="cranfield_ai_services",
        help="Name of the MongoDB database (default: cranfield_ai_services)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="ai_services",
        help="Name of the MongoDB collection (default: ai_services)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file name (default: {DEFAULT_OUTPUT_FILE})"
    )

    args = parser.parse_args()

    if args.option == 1:
        export_mongodb_to_json(args.mongo_uri, args.database, args.collection, args.output)
    else:
        print("Invalid option. Currently, only option 1 is supported.")

if __name__ == "__main__":
    main()