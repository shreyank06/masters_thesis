from collect_data import collect_csv_data
from split_and_predict import split_train_test_data, predict
import json
from datetime import datetime

def main():
    
  with open("processing_configuration.json", "r") as config_file:
      config = json.load(config_file)

  config['start_time'] = datetime.fromisoformat(config['start_time'])
  config['end_time'] = datetime.fromisoformat(config['end_time'])

  if(config['get_csv']):
     collect_csv_data(config['start_time'], config['end_time'], config)

if __name__ == "__main__":
    main()
