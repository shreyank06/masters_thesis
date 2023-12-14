import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Prometheus server details
PROMETHEUS_URL = "http://192.168.254.130:9090/prom/api/v1/query_range"

# Define queries
queries = ['open5G_bt_subscriber_count']

# Define the start time for the queries
start_time = datetime(2023, 12, 14, 10, 28)
step = "0s100ms"

# Maximum number of points allowed per query
max_points = 11000

# Calculate the total duration in milliseconds for each query
total_duration_ms = max_points * 10  # 10 ms per point

# Convert total duration to seconds for timedelta
total_duration_seconds = total_duration_ms / 1000

# Function to fetch data from Prometheus
def fetch_prometheus_data(query, start, end, step):
    params = {
        'query': query,
        'start': start.isoformat() + 'Z',
        'end': end.isoformat() + 'Z',
        'step': step
    }
    response = requests.get(PROMETHEUS_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.text}")

while True:
    # Calculate the end time for this iteration, ensuring it's within the maximum allowed range
    end_time = start_time + timedelta(seconds=total_duration_seconds)

    all_dataframes = []
    for query in queries:
        data = fetch_prometheus_data(query, start_time, end_time, step)
        if data and "data" in data and "result" in data["data"]:
            for i, metric in enumerate(data["data"]["result"]):
                df = pd.DataFrame(metric["values"], columns=['timestamp', f'{query}_{i}'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                all_dataframes.append(df)

    # Check if all_dataframes is not empty
    if all_dataframes:
        # Combine all dataframes
        final_df = pd.concat(all_dataframes, axis=1)

        # Append to CSV (or you can process the DataFrame as needed)
        with open("iterative_data.csv", "a") as f:
            final_df.to_csv(f, header=f.tell()==0)
    else:
        print("No data to concatenate for this iteration.")

    # Update the start time for the next iteration
    start_time = end_time

    # Wait for the specified interval before the next fetch
    # You might want to add a sleep interval here, but it should be less than the query interval
    #time.sleep(total_duration_seconds)
