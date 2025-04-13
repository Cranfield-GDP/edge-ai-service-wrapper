import sys
import os
import json
import time
from pymongo import MongoClient
from rich.console import Console
from rich.table import Table
import traceback

DEFAULT_OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "database_storage", f"cranfield_ai_services__{time.time()}.json"
)
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

def option_export_mongodb_to_json():
    """Export MongoDB data to a JSON file."""
    try:
        uri = input(f"Enter MongoDB URI (default: {DEFAULT_MONGO_URI}): ").strip() or DEFAULT_MONGO_URI
        database_name = input("Enter the database name (default: cranfield_ai_services): ").strip() or "cranfield_ai_services"
        collection_name = input("Enter the collection name (default: ai_services): ").strip() or "ai_services"
        output_file = input(f"Enter the output file path (default: {DEFAULT_OUTPUT_FILE}): ").strip() or DEFAULT_OUTPUT_FILE

        export_mongodb_to_json(uri, database_name, collection_name, output_file)
    except Exception as e:
        print(f"Error: {e}")

OPTIONS = [
    {
        "label": "Export MongoDB data to a JSON file",
        "function": option_export_mongodb_to_json,
    },
]

def main():
    console = Console()

    while True:
        # Create a Rich table for the options
        table = Table(title="Service Manager Options")
        table.add_column("Option", justify="center", style="cyan", no_wrap=True)
        table.add_column("Description", style="yellow")

        for i, option in enumerate(OPTIONS, start=1):
            table.add_row(str(i), option["label"])
            # Add a separator row for better readability
            table.add_row("", "[dim]-----------------------------[/dim]")

        # Add the Quit option at the end
        table.add_row("q", "Quit")

        # Print the table
        console.print(table)

        try:
            choice = input("Enter your choice (1 or 'q' to quit): ").strip()
            
            if choice == "q":
                console.print("[bold green]Exiting the program. Goodbye![/bold green]")
                sys.exit(0)
            elif choice.isdigit() and 1 <= int(choice) <= len(OPTIONS):
                option = OPTIONS[int(choice) - 1]
                console.print(f"[bold yellow]Executing:[/bold yellow] {option['label']}")
                option["function"]()
            else:
                console.print("[bold red]Invalid choice. Please select a valid option (1 or 'q').[/bold red]")
        
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {e}")
            traceback.print_exc()
            console.print("[bold red]Returning to the main menu...[/bold red]")

if __name__ == "__main__":
    main()