from parse_json import load_configuration, fetch_and_convert_data
import time
from datetime import datetime, timedelta
import os

def collect_csv_data(start_time, end_time, config):

    csv_file = "scaled_merged_http_cpu_mem_data.csv"
    
    while True:
        end_time = start_time + timedelta(seconds=60)

        df = fetch_and_convert_data(config, 'queries', start_time, end_time, config['step'])
        df.to_csv(csv_file, index=True, header=True if not os.path.exists(csv_file) else False, mode='a' if os.path.exists(csv_file) else 'w')

        start_time = end_time
        time.sleep(1)  # Sleep for 60 seconds before the next iteration
