import time
import os
import pandas as pd
from datetime import datetime, timedelta
from merge import load_configuration, fetch_and_convert_data, merge_dataframes
from model import createXY
import sys

# import numpy as np
# from matplotlib import pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import GridSearchCV

def main():
    config = load_configuration()
    component = config['component']
    train_percentage = 0.8
    column_types = ['global', 'packet', 'session', 'transaction']
    scaler = MinMaxScaler(feature_range=(0,1))
    csv_file = "scaled_merged_http_cpu_mem_data.csv"

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
        merged_df = merged_df.dropna()
        timestamps = merged_df.index
        scaled_features = scaler.fit_transform(merged_df)
        scaled_df = pd.DataFrame(scaled_features, columns=merged_df.columns, index=timestamps)
        scaled_df.to_csv(csv_file, index=True, header=True if not os.path.exists(csv_file) else False, mode='a' if os.path.exists(csv_file) else 'w')

        print(scaled_df.shape)

        # Calculate the split index
        split_index = int(len(merged_df) * train_percentage)

        # Split the data into training and testing sets
        train_data = scaled_df.iloc[:split_index]
        test_data = scaled_df.iloc[split_index:]

        trainX,trainY=createXY(train_data,5)
        testX,testY=createXY(test_data,5)

        print(trainX.shape, trainY.shape)
        print(trainX[0])
        print(trainY[0])
        # # Convert trainX and trainY to DataFrames
        # trainX_df = pd.DataFrame(trainX.reshape(trainX.shape[0], -1), columns=[f'feature_{i}' for i in range(trainX.shape[1] * trainX.shape[2])])
        # trainY_df = pd.DataFrame(trainY, columns=['target'])
        
        # # Concatenate trainX and trainY horizontally
        # train_combined_df = pd.concat([trainX_df, trainY_df], axis=1)

        # # Save combined trainX and trainY to a single CSV file
        # train_combined_df.to_csv('train_data.csv', index=False)

        #sys.exit()
        #print(trainX, trainY)

        config['start_time'] = end_time
        time.sleep(1)  # Sleep for 60 seconds before the next iteration

if __name__ == "__main__":
    main()
