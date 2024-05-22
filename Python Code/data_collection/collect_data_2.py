import pandas as pd
import matplotlib.pyplot as plt
from .parse_json import fetch_and_convert_data
import time
from datetime import datetime, timedelta
import os
import sys
import re
import plotly.graph_objs as go
from math import isclose

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

        # Initialize variables to store information about the smallest total allocated memory
        smallest_total_allocated_memory = float('inf')
        smallest_mempool_name = ''

        # Filter columns containing "phoenix_memory_allocated" to fetch mempool types
        mempool_columns = [col for col in self.df.columns if 'phoenix_memory_allocated_cm' in col]

        # Iterate over each mempool type
        for mempool_type in mempool_columns:
            mempool_name = mempool_type.split("_", maxsplit=4)[-1]  # Extract mempool name from column name
            total_allocated_memory_col = f'total_allocated_memory_{mempool_name}'

            # Compare total allocated memory and update smallest mempool if needed
            if self.df[total_allocated_memory_col].iloc[-1] < smallest_total_allocated_memory:
                smallest_total_allocated_memory = self.df[total_allocated_memory_col].iloc[-1]
                smallest_mempool_name = mempool_name

        if self.df[f'phoenix_memory_allocated_cm_{smallest_mempool_name}'].nunique() == 1:
            # If they are the same, name it static memory needed for that mempool
            static_memory_needed = self.df[f'phoenix_memory_allocated_cm_{smallest_mempool_name}'].iloc[-1] / 1048576
            
            # Create a new DataFrame for the results
            df_results = pd.DataFrame({
                'Date': [datetime.now().strftime('%Y-%m-%d')],
                f'Static memory needed for {smallest_mempool_name}': [static_memory_needed]
            })
            
            # Save the static memory results directly to the results CSV
            results_csv_filename = f"per_ue_{self.config['component']}_needed.csv"
            results_csv_path = os.path.join('data', results_csv_filename)
            
            if os.path.exists(results_csv_path):
                df_existing_results = pd.read_csv(results_csv_path)
                df_combined_results = pd.concat([df_existing_results, df_results], ignore_index=True)
                df_combined_results.to_csv(results_csv_path, index=False)
            else:
                df_results.to_csv(results_csv_path, index=False)
            
            means = df_combined_results.mean(skipna=True)
            print(means)
            print("Static memory value appended directly to the results CSV.")

            # Add the static memory to mempool_multipliers.csv without date
            multipliers_csv_filename = 'mempool_multipliers.csv'
            multipliers_csv_path = os.path.join('data', multipliers_csv_filename)
            static_memory_needed = float(static_memory_needed)
            
            # Convert means to a DataFrame for easier manipulation, with index reset for appending
            means_df = means.reset_index()
            means_df.columns = ['Metric', 'Value']

            # Check if the CSV file exists
            if os.path.exists(multipliers_csv_path):
                # Load the existing CSV into a DataFrame
                existing_df = pd.read_csv(multipliers_csv_path, header=None, names=['Metric', 'Value'])
                
                # Update or append mean values
                for index, row in means_df.iterrows():
                    metric = row['Metric']
                    value = row['Value']
                    
                    # Check if the metric already exists
                    if metric in existing_df['Metric'].values:
                        # Update the existing row
                        existing_df.loc[existing_df['Metric'] == metric, 'Value'] = value
                    else:
                        # Append the new metric and value
                        existing_df = existing_df.append({'Metric': metric, 'Value': value}, ignore_index=True)
                
                # Save the updated DataFrame back to the CSV file without header
                existing_df.to_csv(multipliers_csv_path, index=False, header=False)
                
            else:
                # Convert the mean to a DataFrame
                means_df = pd.DataFrame(means)
                
                # Save the mean to a new CSV file
                means_df.to_csv(multipliers_csv_path, header=False)
            
            print("Static memory value added directly to mempool_multipliers.csv.")
        
        else:
            # Calculate Per UE memory needed for the smallest total allocated memory mempool
            per_ue_mempool_needed = ((self.df[f'phoenix_memory_allocated_cm_{smallest_mempool_name}'].iloc[-1] / 1048576) +
                                    (smallest_total_allocated_memory - (self.df[f'phoenix_memory_allocated_cm_{smallest_mempool_name}'].iloc[-1] / 1048576))) / latest_ue_count

            # Create a new DataFrame for the results
            df_results = pd.DataFrame({
                'Date': [datetime.now().strftime('%Y-%m-%d')],
                f'Per UE {smallest_mempool_name} mempool needed': [per_ue_mempool_needed]
            })

            # Save the results to a new CSV file with component name
            results_csv_filename = f"per_ue_{self.config['component']}_needed.csv"
            results_csv_path = os.path.join('data', results_csv_filename)
            
            # Check if results CSV file exists
            if os.path.exists(results_csv_path):
                # Load existing results CSV file
                df_existing_results = pd.read_csv(results_csv_path)
                # Convert columns to numeric
                df_existing_results = df_existing_results.apply(pd.to_numeric, errors='coerce')
                df_filled = df_existing_results.fillna(method='ffill')
                mode_values = df_filled.mode().iloc[0]
                df_result = df_filled.apply(lambda col: mode_values[col.name], axis=0)
                print(df_result)
                # Append new results to existing file
                df_combined_results = pd.concat([df_existing_results, df_results], ignore_index=True)
                df_combined_results.to_csv(results_csv_path, index=False)

                # Calculate standard deviation of the new series
                if(len(df_combined_results[f'Per UE {smallest_mempool_name} mempool needed'].dropna()) > 1):
                
                    std_deviation = df_combined_results[f'Per UE {smallest_mempool_name} mempool needed'].dropna().std()
                    rounded_std_deviation = round(std_deviation, 2)
                    if isclose(rounded_std_deviation, 0, abs_tol=1e-3):
                        # Save combined results back to the same CSV file
                        means = df_combined_results.mean(skipna=True)
                        print(means)
                        multipliers_csv_filename = 'mempool_multipliers.csv'
                        multipliers_csv_path = os.path.join('data', multipliers_csv_filename)
                        # Convert means to a DataFrame for easier manipulation, with index reset for appending
                        means_df = means.reset_index()
                        means_df.columns = ['Metric', 'Value']
                        # Check if the CSV file exists
                        if os.path.exists(multipliers_csv_path):
                            # Load the existing CSV into a DataFrame
                            existing_df = pd.read_csv(multipliers_csv_path, header=None, names=['Metric', 'Value'])
                            
                            # Update or append mean values
                            for index, row in means_df.iterrows():
                                metric = row['Metric']
                                value = row['Value']
                                
                                # Check if the metric already exists
                                if metric in existing_df['Metric'].values:
                                    # Update the existing row
                                    existing_df.loc[existing_df['Metric'] == metric, 'Value'] = value
                                else:
                                    # Append the new metric and value
                                    existing_df = existing_df.append({'Metric': metric, 'Value': value}, ignore_index=True)
                            
                            # Save the updated DataFrame back to the CSV file without header
                            existing_df.to_csv(multipliers_csv_path, index=False, header=False)
                        else:
                            # Convert the mean to a DataFrame
                            means_df = pd.DataFrame(means)
                            
                            # Save the mean to a new CSV file
                            means_df.to_csv(multipliers_csv_path, header = False)

                        df_combined_results.to_csv(results_csv_path, index=False)
                        print("Data is symmetric. New value appended.")
                        print(rounded_std_deviation)
                    
                    else:
                        # Remove the newly appended value
                        df_existing_results.to_csv(results_csv_path, index=False)
                        print("Data is not symmetric. New value deleted.")

                else:
                    print(f"add one more data point for 'Per UE {smallest_mempool_name} mempool needed'")

            else:
                # If results CSV file doesn't exist, create a new one
                df_results.to_csv(results_csv_path, index=False)
                print("Results CSV file created.")

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

