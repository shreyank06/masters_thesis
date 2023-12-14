import requests
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta


# Prometheus server details
PROMETHEUS_URL = "http://192.168.254.130:9090/prom/api/v1/query_range"

# Define queries
queries = ["http_client_request_count", "http_client_response_count", 'open5G_bt_subscriber_count']

# # Define the time range for the queries
# start_time = datetime(2023, 12,12, 10, 20)
# end_time = datetime(2023, 12, 12, 10, 35)
# step = "1s"

# Existing start_time and desired step size
start_time = datetime(2023, 12, 14, 10, 27)
step = "0s10ms"

# Maximum number of points allowed
max_points = 11000

# Calculate the total duration in milliseconds
total_duration_ms = max_points * 10  # 10 ms per point

# Convert total duration to seconds (for timedelta)
total_duration_seconds = total_duration_ms / 1000

# Calculate end_time based on start_time and total duration
end_time = start_time + timedelta(seconds=total_duration_seconds)

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

# Fetch data and store in DataFrame
all_dataframes = []
for query in queries:
    data = fetch_prometheus_data(query, start_time, end_time, step)
    if data and "data" in data and "result" in data["data"]:
        for i, metric in enumerate(data["data"]["result"]):
            df = pd.DataFrame(metric["values"], columns=['timestamp', f'{query}_{i}'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            all_dataframes.append(df)

# Combine all dataframes
final_df = pd.concat(all_dataframes, axis=1)

# Print the DataFrame columns
print(final_df.columns)

# Save the DataFrame to a CSV file
csv_file_name = "prometheus_data.csv"
final_df.to_csv(csv_file_name)

print(f"Data saved to {csv_file_name}")
