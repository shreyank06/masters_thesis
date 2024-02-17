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
        csv_files = [f for f in os.listdir(component_folder) if f.endswith('.csv')]
        for csv_file in csv_files:
            series_match = re.search(r'_s(\d+)_', csv_file)
            if series_match:
                series_numbers.append(int(series_match.group(1)))
        return series_numbers

    def collect_csv_data(self):
        while True:
            df = fetch_and_convert_data(self.config, 'queries', self.start_time, self.end_time, self.config['step'])
            latest_subscriber_count = df['subscriber_count_Connected'].iloc[-1]
            #print(latest_subscriber_count)
            # Check if the dataframe has at least 5 rows and the connected column exists
            if len(df) >= 5 and self.connected_col_name in df.columns:
                first_five_values = df[self.connected_col_name].head(10).tolist()
                # Calculate differences between consecutive values
                differences = [first_five_values[i+1] - first_five_values[i] for i in range(len(first_five_values)-1)]
                print(differences)
                if all(first_five_values[i] <= first_five_values[i+1] for i in range(len(first_five_values)-1)):
                    # Values are increasing, adjust filename
                    existing_series_numbers = self.get_existing_series()
                    if existing_series_numbers:
                        new_series = max(existing_series_numbers) + 1
                    else:
                        new_series = 0
                    file_suffix = f"_s{new_series}"
                    component_folder = self.config['component'] + '_csv_files'
                    csv_file_path = os.path.join(component_folder, f"{self.registration_number}_{self.ops_per_second}_{0}{file_suffix}_{self.config['component']}.csv")
                    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
                    df.to_csv(csv_file_path, index=True, header=True if not os.path.exists(csv_file_path) else False, mode='a' if os.path.exists(csv_file_path) else 'w')

            sys.exit()
            # self.start_time = self.end_time
            # time.sleep(1)  # Sleep for 60 sec
