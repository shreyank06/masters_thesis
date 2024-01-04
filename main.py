import time
import os
import pandas as pd
from datetime import datetime, timedelta
from merge import load_configuration, fetch_and_convert_data, merge_dataframes

def main():
    config = load_configuration()
    component = config['component']
    train_percentage = 0.8
    column_types = ['global', 'packet', 'session', 'transaction']
    while True:
        end_time = config['start_time'] + timedelta(seconds=60)

        # Fetch and convert all data
        http_df = fetch_and_convert_data(config, 'http', config['start_time'], end_time, config['step'], column_prefix='http_client_request_count')
        cpu_df = fetch_and_convert_data(config, 'cpu', config['start_time'], end_time, config['step'], column_prefix='cpu')
        allocated_df = fetch_and_convert_data(config, 'allocated', config['start_time'], end_time, config['step'], column_prefix='allocated').apply(pd.to_numeric, errors='coerce')
        wasted_df = fetch_and_convert_data(config, 'wasted', config['start_time'], end_time, config['step'], column_prefix='wasted').apply(pd.to_numeric, errors='coerce')
        subscriber_count_df = fetch_and_convert_data(config, 'subscriber_count', config['start_time'], end_time, config['step'], column_prefix='subscriber_count')
        
        result_df = pd.DataFrame()  # Create an empty DataFrame to store the results

        for column_type in column_types:
            allocated_col = f'phoenix_memory_cm_{column_type}P_allocated_{component}'
            wasted_col = f'phoenix_memory_cm_{column_type}P_wasted_{component}'
            result_df[f'phoenix_memory_{column_type}_used_{component}'] = allocated_df[allocated_col] - wasted_df[wasted_col]

        result_df = result_df.filter(like='_used_amf')

        # Merge HTTP, CPU, Allocated, Wasted and subscriber dataframes
        merged_df = merge_dataframes([http_df, cpu_df, subscriber_count_df, result_df])
        merged_df.to_csv("merged_http_cpu_mem_data_2.csv", index=True, mode='a', header=not os.path.exists("merged_http_cpu_mem_data_2.csv"))

        # Calculate the split index
        split_index = int(len(merged_df) * train_percentage)

        # Split the data into training and testing sets
        train_data = merged_df.iloc[:split_index]
        test_data = merged_df.iloc[split_index:]

        config['start_time'] = end_time
        time.sleep(1)  # Sleep for 60 seconds before the next iteration

        print(f"Training set shape: {train_data.shape}", train_data)
        print(f"Testing set shape: {test_data.shape}")


if __name__ == "__main__":
    main()
