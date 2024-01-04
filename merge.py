import requests
import pandas as pd
from datetime import datetime, timedelta
import json

def load_configuration():
    with open("processing_configuration.json", "r") as config_file:
        config = json.load(config_file)

    config['start_time'] = datetime.fromisoformat(config['start_time'])
    return config

def fetch_prometheus_data(config, query_type, start, end, step):
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

def convert_to_dataframe(config, data, column_prefix='', suffix=''):
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
                    column_name = f'phoenix_memory_{memtype}_{column_prefix}_{component}'
                else:
                    column_name = f'{column_prefix}_{component}'
                df = pd.DataFrame(metric["values"], columns=['timestamp', column_name])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                if not merged_df.empty:
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)
                else:
                    merged_df = df
            if ('open5G_bt_subscriber_count' in str(data)):  
                subscriber_state = metric["metric"].get("subscriber_state", "unknown") 
                column_name = f'subscriber_count_{subscriber_state}'
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

def fetch_and_convert_data(config, query_type, start_time, end_time, step, column_prefix='', suffix=''):
    data = fetch_prometheus_data(config, query_type, start_time, end_time, step)
    return convert_to_dataframe(config, data, column_prefix=column_prefix, suffix=suffix)

