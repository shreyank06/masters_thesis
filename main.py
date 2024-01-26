import time
import os
import pandas as pd
from datetime import datetime, timedelta
from parse_json import load_configuration, fetch_and_convert_data
from model import createXY, build_model, retrain_model
import sys
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from tensorflow.keras.models import Sequential, load_model
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

        df = fetch_and_convert_data(config, 'queries', config['start_time'], end_time, config['step'], column_prefix=config['column_prefix']).apply(pd.to_numeric, errors='coerce')


        # for column_type in column_types:
        #     allocated_col = f'phoenix_memory_cm_{column_type}P_allocated_{component}'
        #     wasted_col = f'phoenix_memory_cm_{column_type}P_wasted_{component}'
        #     result_df[f'phoenix_memory_{column_type}_used_{component}'] = df[allocated_col] - f[wasted_col]

        # scaled_df = pd.DataFrame(scaler.fit_transform(merged_df), columns=merged_df.columns, index=merged_df.index)
        df.to_csv(csv_file, index=True, header=True if not os.path.exists(csv_file) else False, mode='a' if os.path.exists(csv_file) else 'w')
        
        # #correlation
        # #print(scaled_df.corrwith(scaled_df.iloc[:, 3]))

        # # Calculate the split index
        # split_index = int(len(merged_df) * train_percentage)

        # # Split the data into training and testing sets
        # train_data, test_data = scaled_df.iloc[:split_index], scaled_df.iloc[split_index:]
        # trainX, trainY, testX, testY = createXY(train_data, 5) + createXY(test_data, 5)
        
        # print(f"Predicti for time range: {config['start_time']} to {end_time}")

        # if(config["retrain_model"]):
        #     retrain_model(scaled_df, split_index).save("saved_model.h5")

        # try:
        #     prediction = load_model("saved_model.h5").predict(testX)
        # except Exception as e:
        #     print(f"Error during prediction: {e}")
            
        # subscriber_count_predicted_value = scaler.inverse_transform(np.reshape(np.repeat(prediction, 11, axis=-1), (len(prediction), 11)))[:, 3]

        # # Print the reshaped and inverse-transformed predicted values
        # print("Predicted subsriber count -- ", subscriber_count_predicted_value)

        # # original values 
        # original = scaler.inverse_transform(np.reshape(np.repeat(testY, 11, axis=-1), (len(testY), 11)))[:, 3]

        # # Print the original values
        # print("\nOriginal Values -- ", original)
        #sys.exit()
        config['start_time'] = end_time
        time.sleep(1)  # Sleep for 60 seconds before the next iteration

if __name__ == "__main__":
    main()
