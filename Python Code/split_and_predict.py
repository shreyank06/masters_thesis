from model import retrain_model
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class Predictor:
    def __init__(self, df, config):
        self.df = df
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.config = config

    def createXY(self, dataset, n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
            dataX.append(dataset.iloc[i - n_past:i, 0:].values)
            dataY.append(dataset.iloc[i, 5])
        return np.array(dataX), np.array(dataY)

    def split_train_test_data(self, train_percentage):
        scaled_df = pd.DataFrame(self.scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
        split_index = int(len(self.df) * train_percentage)

        train_data, test_data = scaled_df.iloc[:split_index], scaled_df.iloc[split_index:]
        self.trainX, self.trainY, self.testX, self.testY = self.createXY(train_data, 5) + self.createXY(test_data, 5)
        
    def predict_on_test_data(self):
        self.split_train_test_data(0.8)

        if self.config["retrain_model"]:
            retrain_model(self.trainX, self.trainY, self.testX, self.testY).save("../saved_model.h5")

        try:
            prediction = load_model("../saved_model.h5").predict(self.testX)
        except Exception as e:
            print(f"Error during prediction: {e}")

        # print("Shapes before inverse transform:")
        # print("Prediction shape:", prediction.shape)
        # print("Repeated prediction shape:", np.repeat(prediction, 14, axis=-1).shape)
        # print("Reshaped prediction shape:", np.reshape(np.repeat(prediction, 14, axis=-1), (len(prediction), 14)).shape)

        subscriber_count_predicted_value = self.scaler.inverse_transform(np.reshape(np.repeat(prediction, 14, axis=-1), (len(prediction), 14)))[:, 6]

        print("Predicted subscriber count -- ", subscriber_count_predicted_value)

        original = self.scaler.inverse_transform(np.reshape(np.repeat(self.testY, 14, axis=-1), (len(self.testY), 14)))[:, 6]
        print("\nOriginal Values -- ", original)


# # Example usage:
# df = pd.read_csv("scaled_merged_http_cpu_mem_data.csv")  # Load your DataFrame here
# predictor = Predictor(df)
# predictor.predict_on_test_data({"retrain_model": False})  # Example usage of predict_on_test_data method
