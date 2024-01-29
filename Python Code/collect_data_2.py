from parse_json import fetch_and_convert_data
import time
from datetime import datetime, timedelta
import os
import sys
import re

class CsvCollector:
    def __init__(self, start_time, end_time, config):
        self.start_time = start_time
        self.end_time = end_time
        self.config = config
        self.connected_col_name = "subscriber_count_Connected"

    def get_existing_series(self):
        # Get series numbers from existing CSV filenames
        series_numbers = []
        component_folder = self.config['component'] + '_csv_files'
        csv_files = [f for f in os.listdir(component_folder) if f.endswith('.csv')]
        for csv_file in csv_files:
            series_match = re.search(r'_s(\d+)_', csv_file)
            if series_match:
                series_numbers.append(int(series_match.group(1)))
        return series_numbers

    def collect_csv_data(self):
        while True:
            df = fetch_and_convert_data(self.config, 'queries', self.start_time, self.end_time, self.config['step'])
            
            # Check if the dataframe has at least 5 rows and the connected column exists
            if len(df) >= 10 and self.connected_col_name in df.columns:
                first_ten_values = df[self.connected_col_name].head(10).tolist()
                # Calculate differences between consecutive values
                differences = [first_ten_values[i+1] - first_ten_values[i] for i in range(len(first_ten_values)-1)]
                if all(first_ten_values[i] <= first_ten_values[i+1] for i in range(len(first_ten_values)-1)):
                    # Values are increasing, adjust filename
                    existing_series_numbers = self.get_existing_series()
                    if existing_series_numbers:
                        new_series = max(existing_series_numbers) + 1
                    else:
                        new_series = 0
                    file_suffix = f"_s{new_series}"
                    component_folder = self.config['component'] + '_csv_files'
                    csv_file_path = os.path.join(component_folder, f"{len(df)}_{differences[0]}_{0}_{file_suffix}_{self.config['component']}.csv")
                    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            df.to_csv(csv_file_path, index=True, header=True if not os.path.exists(csv_file_path) else False, mode='a' if os.path.exists(csv_file_path) else 'w')

            sys.exit()
            self.start_time = self.end_time
            time.sleep(1)  # Sleep for 60 sec
