import time
import os
import pandas as pd
from datetime import datetime, timedelta
from merge import load_configuration, fetch_and_convert_data, merge_dataframes
from model import createXY, build_model, retrain_model
import sys
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def main():
    config = load_configuration()
    component = config['component']
    train_percentage = 0.8
    column_types = ['global', 'packet', 'session', 'transaction']
    scaler = MinMaxScaler(feature_range=(0,1))
    csv_file = "scaled_merged_http_cpu_mem_data.csv"
      # Create an empty DataFrame to store the results
    while True:
        end_time = config['start_time'] + timedelta(seconds=60)
        result_df = pd.DataFrame()

        # Fetch and convert all data
        http_df = fetch_and_convert_data(config, 'http', config['start_time'], end_time, config['step'], column_prefix='http_client_request_count').apply(pd.to_numeric, errors='coerce')
        cpu_df = fetch_and_convert_data(config, 'cpu', config['start_time'], end_time, config['step'], column_prefix='cpu').apply(pd.to_numeric, errors='coerce')
        allocated_df = fetch_and_convert_data(config, 'allocated', config['start_time'], end_time, config['step'], column_prefix='allocated').apply(pd.to_numeric, errors='coerce')
        wasted_df = fetch_and_convert_data(config, 'wasted', config['start_time'], end_time, config['step'], column_prefix='wasted').apply(pd.to_numeric, errors='coerce')
        subscriber_count_df = fetch_and_convert_data(config, 'subscriber_count', config['start_time'], end_time, config['step'], column_prefix='subscriber_count').apply(pd.to_numeric, errors='coerce')

        for column_type in column_types:
            allocated_col = f'phoenix_memory_cm_{column_type}P_allocated_{component}'
            wasted_col = f'phoenix_memory_cm_{column_type}P_wasted_{component}'
            result_df[f'phoenix_memory_{column_type}_used_{component}'] = allocated_df[allocated_col] - wasted_df[wasted_col]

        result_df = result_df.filter(like='_used_amf')

        # Merge HTTP, CPU, Allocated, Wasted, and subscriber dataframes
        merged_df = merge_dataframes([http_df, cpu_df, subscriber_count_df, result_df])#.dropna()
        timestamps = merged_df.index
        scaled_df = pd.DataFrame(scaler.fit_transform(merged_df), columns=merged_df.columns, index=timestamps)
        merged_df.to_csv(csv_file, index=True, header=True if not os.path.exists(csv_file) else False, mode='a' if os.path.exists(csv_file) else 'w')
        # Calculate the split index
        split_index = int(len(merged_df) * train_percentage)

        # Split the data into training and testing sets
        train_data, test_data = scaled_df.iloc[:split_index], scaled_df.iloc[split_index:]
        trainX, trainY, testX, testY = createXY(train_data, 5) + createXY(test_data, 5)
        
        print(f"Predicting for time range: {config['start_time']} to {end_time}")

        if(config["retrain_model"]):
            retrain_model(trainX, trainY, testX, testY).save("saved_model.h5")
        try:
            prediction = load_model("saved_model.h5").predict(testX)
        except Exception as e:
            print(f"Error during prediction: {e}")
            
        subscriber_count_predicted_value = scaler.inverse_transform(np.reshape(np.repeat(prediction, 11, axis=-1), (len(prediction), 11)))[:, 3]

        # Print the reshaped and inverse-transformed predicted values
        print("Predicted subsriber count -- ", subscriber_count_predicted_value)

        # original values 
        original = scaler.inverse_transform(np.reshape(np.repeat(testY, 11, axis=-1), (len(testY), 11)))[:, 3]

        # Print the original values
        print("\nOriginal Values -- ", original)
        #sys.exit()
        config['start_time'] = end_time
        time.sleep(1)  # Sleep for 60 seconds before the next iteration

if __name__ == "__main__":
    main()
