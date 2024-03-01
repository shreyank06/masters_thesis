import pandas as pd
import matplotlib.pyplot as plt
from parse_json import fetch_and_convert_data
import time
from datetime import datetime, timedelta
import os
import sys
import re
import plotly.graph_objs as go

class CsvCollector:
    def __init__(self, start_time, end_time, config, registration_number, ops_per_second):
        self.start_time = start_time
        self.end_time = end_time
        self.config = config
        self.connected_col_name = "subscriber_count_Connected"
        self.registration_number = registration_number
        self.ops_per_second = ops_per_second
        self.df = None

    def get_existing_series(self):
        # Get series numbers from existing CSV filenames
        series_numbers = []
        component_folder = self.config['component'] + '_new_csv_files_2'
        component_folder_path = os.path.join(os.getcwd(), component_folder)
        if not os.path.exists(component_folder_path):
            os.makedirs(component_folder_path)
        csv_files = [f for f in os.listdir(component_folder) if f.endswith('.csv')]
        for csv_file in csv_files:
            series_match = re.search(r'_s(\d+)_', csv_file)
            if series_match:
                series_numbers.append(int(series_match.group(1)))
        return series_numbers, component_folder_path
    
    def fetch_and_save_csv(self):
        # Check if 'data' folder exists, if not, create it
        if not os.path.exists('data'):
            os.makedirs('data')

        csv_filename = f"{self.config['component']}_{self.registration_number}_{self.ops_per_second}_set.csv"
        
        counter = 0
        while os.path.exists(os.path.join('data', csv_filename)):
            counter += 1
            csv_filename = f"{self.config['component']}_{self.registration_number}_{self.ops_per_second}_set_{counter}.csv"

        # Save CSV file in 'data' folder
        self.df.to_csv(os.path.join('data', csv_filename), index=True)
        # self.visualize_data(self.df, 'data', '')  # Assuming this line is a visualization function call
        
        print(f"CSV file saved as: {csv_filename} in 'data' folder.")

    def collect_csv_data(self):
            self.df = fetch_and_convert_data(self.config, 'queries', self.start_time, self.end_time, self.config['step'])
            self.fetch_and_save_csv()

            # existing_series_numbers,component_folder_path = self.get_existing_series()
            # if existing_series_numbers:
            #     new_series = max(existing_series_numbers) + 1
            # else:
            #     new_series = 0
            # file_suffix = f"_s{new_series}"

            # # csv_file_path = os.path.join(component_folder_path, f"{self.registration_number}_{self.ops_per_second}_{0}{file_suffix}_{self.config['component']}.csv")
            # # self.df.to_csv(csv_file_path, index=True, header=True if not os.path.exists(csv_file_path) else False, mode='a' if os.path.exists(csv_file_path) else 'w')

            # self.visualize_data(self.df, component_folder_path, file_suffix)

            # sys.exit()
            # self.start_time = self.end_time
            # time.sleep(1)  # Sleep for 60 sec

    def visualize_data(self, df, folder_path, file_suffix):
        index_column = df.index

        num_columns = len(df.columns)

        # Create an empty figure
        fig = go.Figure()

        # Add traces for each column
        for col in df.columns[0:]:  # Start from the second column
            fig.add_trace(go.Scatter(x=index_column, y=df[col], mode='lines', name=col))

        # Update layout
        fig.update_layout(
            title=f'All Columns vs {index_column}',
            xaxis_title=str(index_column),  # Convert DatetimeIndex to string
            yaxis_title='Values',
            autosize=False,
            width=1000,
            height=600,
            # Enable interactive legend for column filtering
            legend=dict(traceorder="normal")
        )


        # Save the plot as an HTML file
        html_file_path = os.path.join(folder_path, f"{self.registration_number}_{self.ops_per_second}_{0}{file_suffix}_all_columns_vs_index.html")
        fig.write_html(html_file_path)

        # Close the figure
        #fig.close()

