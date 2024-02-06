from collect_data import CsvCollector
import json
from datetime import datetime
import pandas as pd
from split_and_predict import Predictor
import concurrent.futures

def process_data(df, config):
    predictor = Predictor(df.apply(pd.to_numeric, errors='coerce'), config)
    predictor.predict_on_test_data()

def main():
    
    with open("../processing_configuration.json", "r") as config_file:
        config = json.load(config_file)

    if config['get_csv']:
        config['start_time'] = datetime.fromisoformat(config['start_time'])
        config['end_time'] = datetime.fromisoformat(config['end_time'])

        collector = CsvCollector(config['start_time'], config['end_time'], config)
        collector.collect_csv_data()

    # Load your DataFrames here in a list
    dfs = []
    dfs.append(pd.read_csv("amf_csv_files/1001_1_0_s0_amf.csv"))
    dfs.append(pd.read_csv("amf_csv_files/1002_1_0_s0_amf.csv"))  # Add more DataFrames as needed

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_data, df, config) for df in dfs]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
