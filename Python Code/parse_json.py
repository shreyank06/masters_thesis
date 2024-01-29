import requests
import pandas as pd
import csv

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
        elif job == "phoenix" and component == 'bt' and result['metric'].get('component')=='bt5g':
            filtered_results.append(result)
        elif job == "process" and config['component'] in result["metric"].get("groupname", ""):
            filtered_results.append(result)
    return filtered_results

import csv

def convert_to_csv(config, data, output_file):
    header_written = False
    
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for metric in filtered_json(config, data):
            job = metric['metric'].get('job', 'unknown')
            component = config['component']
            if job == 'phoenix':
                if 'http' in str(metric):
                    metric_name = metric["metric"].get("__name__", "unknown")
                    column_name = f'{metric_name}_{component}'
                if 'allocated' in str(metric):
                    memtype = metric["metric"].get("memtype", "unknown")
                    column_name = f'phoenix_memory_allocated_{memtype}_{component}'
                if 'wasted' in str(metric):
                    memtype = metric["metric"].get("memtype", "unknown")
                    column_name = f'phoenix_memory_wasted_{memtype}_{component}'
                if 'open5G_bt_subscriber_count' in str(metric):  
                    subscriber_state = metric["metric"].get("subscriber_state", "unknown") 
                    column_name = f'subscriber_count_{subscriber_state}'
            if job == 'process':
                if 'cpu' in str(metric):
                    mode = metric["metric"].get("mode", "unknown")
                    column_name = f'cpu_{component}_{mode}'
                    
            values = [(entry['timestamp'], entry['value']) for entry in metric["values"]]
            
            if not header_written:
                csv_writer.writerow(['timestamp', column_name])
                header_written = True
                
            csv_writer.writerows(values)
    print(values)

def convert_to_dataframe(config, data):

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
            if 'allocated' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_allocated_{memtype}_{component}'
            if'wasted' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_wasted_{memtype}_{component}'
            if ('open5G_bt_subscriber_count' in str(metric)):  
                subscriber_state = metric["metric"].get("subscriber_state", "unknown") 
                column_name = f'subscriber_count_{subscriber_state}'
        if job == 'process':
            if'cpu' in str(metric):
                mode = metric["metric"].get("mode", "unknown")
                column_name = f'cpu_{component}_{mode}'
        df = pd.DataFrame(metric["values"], columns=['timestamp', column_name])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        if not merged_df.empty:
            merged_df = pd.merge(merged_df, df, how='inner', left_index=True, right_index=True)
        else:
            merged_df = df
    
    merged_df = merged_df.apply(pd.to_numeric, errors='ignore')

    for memtype in ['cm_globalP', 'cm_packetP', 'cm_sessionP', 'cm_transactionP']:
        alloc_col = f'phoenix_memory_allocated_{memtype}_{component}'
        waste_col = f'phoenix_memory_wasted_{memtype}_{component}'
        if all(col in merged_df for col in [alloc_col, waste_col]):
            merged_df[f'phoenix_memory_used_{memtype}_{component}'] = merged_df.pop(alloc_col) - merged_df.pop(waste_col)
            
    return merged_df


def fetch_and_convert_data(config, query_type, start_time, end_time, step):
    data = fetch_prometheus_data(config, start_time, end_time, step)
    return convert_to_dataframe(config, data)