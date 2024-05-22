import os
import pandas as pd
import json
import logging
from flask import Flask, send_from_directory, jsonify

app = Flask(__name__)
app.debug = True  # Enable debug mode

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Ensure Flask uses the same logger
app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)

def extract_component(index):
    try:
        # Extract the component from the index after "Per UE"
        components = index.split('_')
        per_ue_index = components.index('Per') if 'Per' in components else -1
        component_string = components[per_ue_index + 2]
        return component_string.split()[0]
    except Exception as e:
        logger.error(f"Error in extract_component with index {index}: {e}")
        return None


def convert_csv_to_json():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file_path = os.path.join(base_dir, 'data', 'mempool_multipliers.csv')
        logger.info(f"CSV file path: {csv_file_path}")

        if not os.path.exists(csv_file_path):
            logger.error(f"CSV file not found at {csv_file_path}")
            return None

        df = pd.read_csv(csv_file_path, index_col=0)
        logger.info(f"CSV file read successfully")

        json_data = {}
        for index, row in df.iterrows():
            component = extract_component(index)
            if component is None:
                logger.warning(f"Skipping row with index {index} due to extraction error")
                continue
            if component not in json_data:
                json_data[component] = {}
            json_data[component][index] = row['Mean']

        json_file_path = os.path.join(base_dir, 'data', 'mempool_multipliers.json')
        logger.info(f"JSON file path: {json_file_path}")

        if os.path.exists(json_file_path):
            logger.warning(f"JSON file already exists at {json_file_path}. Overwriting...")

        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        logger.info(f"JSON file saved at {json_file_path}")

        return json_file_path
    except Exception as e:
        logger.error(f"Error in convert_csv_to_json: {e}")
        return None

@app.route('/mempool_multipliers')
def serve_json():
    try:
        json_file_path = convert_csv_to_json()
        if json_file_path:
            logger.info(f"Serving JSON file from {json_file_path}")
            return send_from_directory(os.path.dirname(json_file_path), os.path.basename(json_file_path))
        else:
            logger.error("Error: JSON file could not be created")
            return jsonify(error="JSON file could not be created"), 500
    except Exception as e:
        logger.error(f"Error in serve_json: {e}")
        return jsonify(error="Internal Server Error"), 500

if __name__ == "__main__":
    try:
        logger.info("Starting Flask application")
        app.run(debug=True, port=5001)
    except Exception as e:
        logger.error(f"Error starting Flask application: {e}")
