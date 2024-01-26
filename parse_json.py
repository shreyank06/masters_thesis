import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import sys

def load_configuration():
    with open("processing_configuration.json", "r") as config_file:
        config = json.load(config_file)

    config['start_time'] = datetime.fromisoformat(config['start_time'])
    config['end_time'] = datetime.fromisoformat(config['end_time'])
    return config

def fetch_prometheus_data(config, start, end, step):
    queries = '|'.join(config['queries'])
    query = '({__name__=~"' + queries + '"})'

    params = {
        'query': query,
        'start': start.isoformat() + 'Z',
        'end': end.isoformat() + 'Z',
        'step': step
    }

    url = config['PROMETHEUS_URL']
    
    response = requests.get(url, params=params)
    #print(response.json())
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.text}")
    
def filtered_json(config, data):
    filtered_results = []
    for result in data["data"]["result"]:
        job = result["metric"].get("job")
        component = result["metric"].get("phnf")
        if job == "phoenix" and component == config['component']:
            filtered_results.append(result)
        elif job == "process" and config['component'] in result["metric"].get("groupname", ""):
            filtered_results.append(result)
    return filtered_results

def convert_to_dataframe(config, data, column_prefix='', suffix=''):
    merged_df = pd.DataFrame()
    
    # Filter the results based on the job type and component
    filtered_results = filtered_json(config, data)
    component = config['component']
    
    for i, metric in enumerate(filtered_results):
        job = metric['metric'].get('job', 'unknown')
        if job == 'phoenix':
            if'http' in str(metric):
                metric_name = metric["metric"].get("__name__", "unknown")
                column_name = f'{metric_name}_{component}'
        if job == 'process':
            if'cpu' in str(metric):
                if 'memory' in str(data):
                    memtype = metric["metric"].get("memtype", "unknown")
                    column_name = f'phoenix_memory_{memtype}_{column_prefix}_{component}'
                # else:
                #     mode = metric["metric"].get("mode", "unknown")
                #     column_name = f'cpu_{component}_{mode}'
        df = pd.DataFrame(metric["values"], columns=['timestamp', column_name])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        if not merged_df.empty:
            merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)
        else:
            merged_df = df
    #print(merged_df)
    #print(merged_df.iloc[:, 2])
    return merged_df


def fetch_and_convert_data(config, query_type, start_time, end_time, step, column_prefix='', suffix=''):
    data = fetch_prometheus_data(config, start_time, end_time, step)
    return convert_to_dataframe(config, data, column_prefix=column_prefix, suffix=suffix)
