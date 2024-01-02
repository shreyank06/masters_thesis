import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
import json

with open("processing_configuration.json", "r") as config_file:
    config = json.load(config_file)

config['start_time'] = datetime.fromisoformat(config['start_time'])

def fetch_prometheus_data(query_type, start, end, step):
    query = config['queries'][query_type][0]
    params = {
        'query': query,
        'start': start.isoformat() + 'Z',
        'end': end.isoformat() + 'Z',
        'step': step
    }
    response = requests.get(config['PROMETHEUS_URL'], params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching {query_type} data: {response.text}")

def convert_to_dataframe(data, column_prefix='', suffix=''):
    merged_df = pd.DataFrame()
    if "data" in data and "result" in data["data"]:
        for i, metric in enumerate(data["data"]["result"]):
            job = next((entry['metric'].get('job', 'unknown') for entry in data["data"]["result"] if 'job' in entry['metric']), 'unknown')
            if(job == 'phoenix'):
                component = metric["metric"].get("phnf", "unknown")
            elif(job == 'process'):
                component = metric["metric"].get("groupname", "unknown").split('/')[-1].split('.')[0]
            if(config['component'] == component):
                if 'cpu' in str(data):
                    mode = metric["metric"].get("mode", "unknown")
                    column_name = f'{column_prefix}_{component}_{mode}'
                elif'memory' in str(data):
                    memtype = metric["metric"].get("memtype", "unknown")
                    column_name = f'phoenix_memory_{memtype}_{column_prefix}_amf'
                else:
                    column_name = f'http_client_request_count_{component}'
                df = pd.DataFrame(metric["values"], columns=['timestamp', column_name])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                if not merged_df.empty:
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)
                else:
                    merged_df = df
    return merged_df

def merge_dataframes(dataframes):
    merged_df = pd.DataFrame()
    for df in dataframes:
        if not merged_df.empty:
            merged_df = pd.merge(merged_df, df, how='left', on='timestamp')
        else:
            merged_df = df
    return merged_df

def fetch_and_convert_data(query_type, start_time, end_time, step, column_prefix='', suffix=''):
    data = fetch_prometheus_data(query_type, start_time, end_time, step)
    return convert_to_dataframe(data, column_prefix=column_prefix, suffix=suffix)

while True:
    end_time = config['start_time'] + timedelta(seconds=60)

    # Fetch and convert HTTP data
    http_df = fetch_and_convert_data('http', config['start_time'], end_time, config['step'], column_prefix='http_client_request_count')
    
    # Fetch and convert CPU data
    cpu_df = fetch_and_convert_data('cpu', config['start_time'], end_time, config['step'], column_prefix='cpu')

    #Fetch and convert memory data
    allocated_df = fetch_and_convert_data('allocated', config['start_time'], end_time, config['step'], column_prefix='allocated', suffix='allocated_amf').apply(pd.to_numeric, errors='coerce')
    wasted_df = fetch_and_convert_data('wasted', config['start_time'], end_time, config['step'], column_prefix='wasted', suffix='wasted_amf').apply(pd.to_numeric, errors='coerce')

    column_types = ['global', 'packet', 'session', 'transaction']
    result_df = wasted_df.copy()
    
    for column_type in column_types:
        allocated_col = f'phoenix_memory_cm_{column_type}P_allocated_amf'
        wasted_col = f'phoenix_memory_cm_{column_type}P_wasted_amf'
        result_df[f'phoenix_memory_{column_type}_used_amf'] = allocated_df[allocated_col] - wasted_df[wasted_col]

    # # Merge HTTP, CPU, Allocated, and Wasted DataFrames
    merged_df = merge_dataframes([http_df, cpu_df, result_df])
    merged_df.to_csv("merged_http_cpu_mem_data_2.csv", index=True, mode='a', header=not os.path.exists("merged_http_cpu_mem_data_2.csv"))

    config['start_time'] = end_time
    time.sleep(1)  # Sleep for 60 seconds before the next iteration
