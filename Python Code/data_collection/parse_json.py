import requests
import pandas as pd
import csv
import sys
import json
from sklearn.feature_extraction import DictVectorizer

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
    results = response.json()['data']['result']

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.text}")


def convert_json_to_features(json):
    vectorizer = DictVectorizer()
    features_list = []

    for result in json['data']['result']:
        metric = result.get('metric', {})
        values = result.get('values', [])  # Get the 'metric' dictionary or an empty dictionary if it doesn't exist
        metric_name = metric.get('__name__', 'N/A')  # Get the 'name' key or set default value to 'N/A' if it doesn't exist
        if metric.get('phnf') == 'smf':
            # print("Metric Name:", metric_name)
            # print("Values:", values)
            features = {}
            features['metric_name'] = metric_name
            features['Values'] = [item[1] for item in values]
            features['index'] = [item[0] for item in values]
            features_list.append(features)
    
    print(features_list)
    X = vectorizer.transform(features_list)
    print(X)
    sys.exit()

def filtered_json(config, data):
    filtered_results = []
    for result in data["data"]["result"]:
        job = result["metric"].get("job")
        component = result["metric"].get("phnf")
        if job == "phoenix" and component == config['component']:
            filtered_results.append(result)
        elif job == "phoenix" and component == 'bt' and result['metric'].get('component')=='bt5g':
            filtered_results.append(result)
        elif job == "process" and config['component'] in result["metric"].get("groupname", ""):
            filtered_results.append(result)
    return filtered_results


def used_memory(merged_df, component):
    for memtype in ['cm_globalP', 'cm_packetP', 'cm_sessionP', 'cm_transactionP']:
        alloc_col = f'phoenix_memory_allocated_{memtype}_{component}'
        waste_col = f'phoenix_memory_wasted_{memtype}_{component}'
        if all(col in merged_df for col in [alloc_col, waste_col]):
            merged_df[f'phoenix_memory_used_{memtype}_{component}'] = merged_df.pop(alloc_col).div(1000000) - merged_df.pop(waste_col).div(1000000)
    return merged_df

def memory_per_ue(merged_df, component):
    for memtype in ['cm_globalP', 'cm_packetP', 'cm_sessionP', 'cm_transactionP']:
        used_col = f'phoenix_memory_used_{memtype}_{component}'
        subscriber_count_col = f'subscriber_count_Connected'
        if all(col in merged_df for col in [used_col, subscriber_count_col]):
            merged_df[f'memory_per_ue_{memtype}_{component}'] = merged_df[used_col] / merged_df[subscriber_count_col]
    return merged_df

def process_memory_to_mb(merged_df, component):
    # Find columns matching the pattern 'process_memory_{component}_*'
    process_memory_cols = [col for col in merged_df.columns if col.startswith(f'process_memory_{component}_')]

    # Convert values in matching columns to megabytes
    for col in process_memory_cols:
        merged_df[col] = merged_df[col] / 1000000

    return merged_df

def convert_to_dataframe(config, data):

    merged_df = pd.DataFrame()
    # Filter the results based on the job type and component
    filtered_results = filtered_json(config, data)
    component = config['component']

    for i, metric in enumerate(filtered_results):
        job = metric['metric'].get('job', 'unknown')
        if job == 'phoenix':
            if 'http' in str(metric):
                metric_name = metric["metric"].get("__name__", "unknown")
                column_name = f'{metric_name}_{component}'
            if 'allocated' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_allocated_{memtype}_{component}'
            if "chunksize" in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_chunksize_{memtype}_{component}'
            if "chunk_count_total" in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_chunk_count_{memtype}_{component}'
            if 'wasted' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_wasted_{memtype}_{component}'
            if 'open5G_bt_subscriber_count' in str(metric):
                subscriber_state = metric["metric"].get("subscriber_state", "unknown")
                column_name = f'subscriber_count_{subscriber_state}'
            if 'allocation_count_total' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_cm_allocation_count_total_{memtype}_{component}'
            if 'max_used_chunks_per_pool_count' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_cm_max_used_chunks_per_pool_count_{memtype}_{component}'
            if 'phoenix_memory_cm_used_chunk_count' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_cm_used_chunk_count_{memtype}_{component}'

        if job == 'process':
            if 'cpu' in str(metric):
                mode = metric["metric"].get("mode", "unknown")
                column_name = f'cpu_{component}_{mode}'
            if "namedprocess_namegroup_memory_bytes" in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'process_memory_{component}_{memtype}'
        df = pd.DataFrame(metric["values"], columns=['timestamp', column_name])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        if not merged_df.empty:
            merged_df = pd.merge(merged_df, df, how='inner', left_index=True, right_index=True)
        else:
            merged_df = df

    merged_df = merged_df.apply(pd.to_numeric, errors='ignore')
    merged_df = multiply_columns(merged_df, component)

    # merged_df = used_memory(merged_df, component)
    # merged_df = memory_per_ue(merged_df, component)
    # merged_df = process_memory_to_mb(merged_df, component)

    return merged_df


def multiply_columns(df, component):
    # Find all columns with 'phoenix_memory_chunksize_' or 'phoenix_memory_chunk_count_' prefix
    columns_to_multiply = [col for col in df.columns if col.startswith(f'phoenix_memory_chunksize_') or col.startswith(f'phoenix_memory_chunk_count_')]
    
    # Print the columns present in the dataframe before multiplication
    for col in columns_to_multiply:
        memtype = col.split('_')[-2]  # Extract memtype from column name
        matching_columns = [c for c in df.columns if memtype in c and (c.startswith(f'phoenix_memory_chunksize_') or c.startswith(f'phoenix_memory_chunk_count_'))]  # Find all columns with the same memtype
        
        if len(matching_columns) > 1:
            total_column_name = f'total_allocated_memory_{memtype}_{component}'
            
            # Check if all matching columns exist in the dataframe
            if all(col in df.columns for col in matching_columns):
                df[total_column_name] = df[matching_columns].prod(axis=1)/1048576  # Multiply columns with the same memtype and then divide by 1048576
                df.drop(columns=matching_columns, inplace=True)  # Remove original columns
            else:
                print(f"Error: Columns {matching_columns} do not exist in the dataframe.")
        
    return df

def fetch_and_convert_data(config, query_type, start_time, end_time, step):
    data = fetch_prometheus_data(config, start_time, end_time, step)
    if config['convert_json_to_features']:
        features = convert_json_to_features(data)

    return convert_to_dataframe(config, data)
