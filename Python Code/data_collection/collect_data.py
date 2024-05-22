import pandas as pd
import matplotlib.pyplot as plt
from .parse_json import fetch_and_convert_data
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

        # Save original CSV file in 'data' folder
        self.df.to_csv(os.path.join('data', csv_filename), index=True)
        
        # Get the latest UE count information
        latest_ue_count = self.df['subscriber_count_Connected'].iloc[-1]

        # Initialize a dictionary to store per UE memory needed for each mempool
        per_ue_mempool_needed = {}

        # Filter columns containing "phoenix_memory_allocated" to fetch mempool types
        mempool_columns = [col for col in self.df.columns if 'phoenix_memory_allocated' in col]
        #print(mempool_columns)

                # Iterate over each mempool type
        for mempool_type in mempool_columns:
            mempool_name = mempool_type.split("_", maxsplit=4)[-1]  # Extract mempool name from column name
            total_allocated_memory_col = f'total_allocated_memory_{mempool_name}'
            #print(total_allocated_memory_col)
            # Calculate Per UE memory needed for the current mempool
            per_ue_mempool_needed[mempool_type] = ((self.df[mempool_type].iloc[-1] / 1048576) + 
                                                    ((self.df[total_allocated_memory_col].iloc[-1]) - (self.df[mempool_type].iloc[-1] / 1048576))) / latest_ue_count
            print(self.df[total_allocated_memory_col].iloc[-1] / 1048576)
            print(self.df[mempool_type])
        sys.exit()
        # Create a new DataFrame for the results
        df_results = pd.DataFrame({
            'Date': [datetime.now().strftime('%Y-%m-%d')],
        })
        # Add per UE memory needed for each mempool to the results DataFrame
        for mempool_type, per_ue_memory in per_ue_mempool_needed.items():
            mempool_name = mempool_type.split("_")[4]  # Extract mempool name from column name
            df_results[f'Per UE {mempool_name} mempool needed'] = [per_ue_memory]
        # Save the results to a new CSV file with component name
        results_csv_filename = f"per_ue_{self.config['component']}_needed.csv"
        results_csv_path = os.path.join('data', results_csv_filename)
        
        # Check if results CSV file exists
        if os.path.exists(results_csv_path):
            # Load existing results CSV file
            df_existing_results = pd.read_csv(results_csv_path)
            # Append new results to existing file
            df_combined_results = pd.concat([df_existing_results, df_results], ignore_index=True)
            # Save combined results back to the same CSV file
            df_combined_results.to_csv(results_csv_path, index=False)
        else:
            # If results CSV file doesn't exist, create a new one
            df_results.to_csv(results_csv_path, index=False)

        print(f"Original CSV file saved as: {csv_filename} in 'data' folder.")
        print(f"Results CSV file saved as: {results_csv_filename} in 'data' folder.")





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

