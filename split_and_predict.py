from model import createXY, retrain_model
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def split_train_test_data(df, train_percentage,scaled_df):

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    # Calculate the split index
    split_index = int(len(df) * train_percentage)
            
    #correlation
    print(scaled_df.corrwith(scaled_df.iloc[:, 4]))

    # Split the data into training and testing sets
    train_data, test_data = scaled_df.iloc[:split_index], scaled_df.iloc[split_index:]
    trainX, trainY, testX, testY = createXY(train_data, 5) + createXY(test_data, 5)
    
def predict_on_test_data(config, scaled_df, split_index, scaler, testX, testY):

    if(config["retrain_model"]):
        retrain_model(scaled_df, split_index).save("saved_model.h5")

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

def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset.iloc[i - n_past:i, 0:].values)
        dataY.append(dataset.iloc[i, 3])
    return np.array(dataX), np.array(dataY)
