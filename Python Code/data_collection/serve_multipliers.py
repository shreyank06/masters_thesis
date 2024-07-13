import os
import csv
import json
import logging
import re
import time
from flask import Flask, jsonify, request, Response
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)

def configure_logging():
    # Configure Flask's logger to log at INFO level
    app.logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

configure_logging()

def convert_csv_to_json(csv_file_path):
    try:
        # Initialize a dictionary to hold the data
        data = {}

        # Read and parse the CSV file
        with open(csv_file_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) != 2:
                    continue  # Skip rows that don't have exactly 2 columns
                key, value = row
                component, mem_type = extract_component_and_type(key)
                if component and mem_type:
                    if mem_type not in data:
                        data[mem_type] = {}
                    data[mem_type][key] = float(value)

        return data

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return None

def extract_component_and_type(key):
    # This function extracts the component and type from the key
    match = re.search(r'(\w+)P_(\w+)', key)
    if match:
        return match.group(1), match.group(2)
    else:
        return None, None

class CSVHandler(FileSystemEventHandler):
    def __init__(self, callback, csv_file_path):
        super().__init__()
        self.callback = callback
        self.csv_file_path = csv_file_path

    def on_modified(self, event):
        if event.src_path == self.csv_file_path:
            app.logger.info("CSV file modified. Re-running the script...")
            self.callback()


def run_script():
    # Get the directory where the current script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one directory back from the base directory
    base_dir = os.path.dirname(base_dir)
    # Construct the path to the CSV file relative to the updated base directory
    csv_file_path = os.path.join(base_dir, 'data', 'mempool_multipliers.csv')
    app.logger.info(f"CSV file path: {csv_file_path}")

    if not os.path.exists(csv_file_path):
        app.logger.error(f"CSV file not found at {csv_file_path}")
        error_response = {"error": "CSV file not found"}
        return Response(json.dumps(error_response), status=404, mimetype='application/json')

    json_data = convert_csv_to_json(csv_file_path)
    if json_data is not None:
        return Response(json.dumps(json_data), status=200, mimetype='application/json')
    else:
        error_response = {"error": "An error occurred while processing the CSV file"}
        return Response(json.dumps(error_response), status=500, mimetype='application/json')



@app.route('/api/mempool_multipliers', methods=['GET'])
def get_mempool_multipliers():
    return run_script()

if __name__ == "__main__":
    # Get the directory where the current script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one directory back from the base directory
    base_dir = os.path.dirname(base_dir)
    # Construct the path to the CSV file relative to the updated base directory
    csv_file_path = os.path.join(base_dir, 'data', 'mempool_multipliers.csv')

    observer = Observer()
    observer.schedule(CSVHandler(run_script, csv_file_path), path=os.path.dirname(csv_file_path), recursive=False)
    observer.start()
    try:
        app.run(port=5001, debug=True)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
