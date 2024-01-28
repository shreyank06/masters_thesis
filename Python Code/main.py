from collect_data import collect_csv_data
import json
from datetime import datetime
import pandas as pd
from split_and_predict import Predictor

def main():
    
    with open("../processing_configuration.json", "r") as config_file:
        config = json.load(config_file)

    config['start_time'] = datetime.fromisoformat(config['start_time'])
    config['end_time'] = datetime.fromisoformat(config['end_time'])

    if(config['get_csv']):
        collect_csv_data(config['start_time'], config['end_time'], config)

    # Example usage:
    df = pd.read_csv("../scaled_merged_http_cpu_mem_data.csv")  # Load your DataFrame here
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    #print(df)

    predictor = Predictor(df.apply(pd.to_numeric, errors='coerce'), config)
    predictor.predict_on_test_data()  # Example usage of predict_on_test_data method

if __name__ == "__main__":
    main()
