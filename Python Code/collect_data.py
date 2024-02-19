from parse_json import fetch_and_convert_data
import time
from datetime import datetime, timedelta
import os
import sys
import re

class CsvCollector:
    def __init__(self, start_time, end_time, config, registration_number, ops_per_second):
        self.start_time = start_time
        self.end_time = end_time
        self.config = config
        self.connected_col_name = "subscriber_count_Connected"
        self.registration_number = registration_number
        self.ops_per_second = ops_per_second

    def get_existing_series(self):
        # Get series numbers from existing CSV filenames
        series_numbers = []
        component_folder = self.config['component'] + '_csv_files'
        component_folder_path = os.path.join(os.getcwd(), component_folder)
        if not os.path.exists(component_folder_path):
            os.makedirs(component_folder_path)
        csv_files = [f for f in os.listdir(component_folder) if f.endswith('.csv')]
        for csv_file in csv_files:
            series_match = re.search(r'_s(\d+)_', csv_file)
            if series_match:
                series_numbers.append(int(series_match.group(1)))
        return series_numbers, component_folder_path

    def collect_csv_data(self):
        while True:
            df = fetch_and_convert_data(self.config, 'queries', self.start_time, self.end_time, self.config['step'])
            existing_series_numbers,component_folder_path = self.get_existing_series()
            if existing_series_numbers:
                new_series = max(existing_series_numbers) + 1
            else:
                new_series = 0
            file_suffix = f"_s{new_series}"

            csv_file_path = os.path.join(component_folder_path, f"{self.registration_number}_{self.ops_per_second}_{0}{file_suffix}_{self.config['component']}.csv")
            df.to_csv(csv_file_path, index=True, header=True if not os.path.exists(csv_file_path) else False, mode='a' if os.path.exists(csv_file_path) else 'w')

            sys.exit()
            # self.start_time = self.end_time
            # time.sleep(1)  # Sleep for 60 sec
