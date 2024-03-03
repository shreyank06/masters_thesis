from model import retrain_model
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from .window_generator import WindowGenerator
import tensorflow as tf
from .baseline import Baseline
from .models import Models
import matplotlib.pyplot as plt

class Predictor:
    def __init__(self, df, config):
        self.df = df
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.config = config
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.scaled_df = None
        self.input_width = None
        self.label_width = None
        self.shift = None
        self.label_columns = None

    def createXY(self, dataset, n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
            dataX.append(dataset.iloc[i - n_past:i, 0:].values)
            dataY.append(dataset.iloc[i, 6])
        return np.array(dataX), np.array(dataY)

    def split_scaled_train_test_data(self, train_percentage):

        scaled_df = pd.DataFrame(self.scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
        split_index = int(len(self.df) * train_percentage)
        train_data, test_data = scaled_df.iloc[:split_index], scaled_df.iloc[split_index:]
        self.trainX, self.trainY, self.testX, self.testY = self.createXY(train_data, 5) + self.createXY(test_data, 5)

    def split_descaled_train_test_data(self, train_percentage):

        split_index = int(len(self.df) * train_percentage)        
        train_data, test_data = self.df.iloc[:split_index], self.df.iloc[split_index:]
        self.trainX, self.trainY, self.testX, self.testY = self.createXY(train_data, 5) + self.createXY(test_data, 5)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        trainX_flattened = self.trainX.reshape(self.trainX.shape[0], -1)
        trainY_flattened = self.trainY.reshape(self.trainX.shape[0], -1)
        np.savetxt(os.path.join(script_dir, 'trainX_descaled.csv'), trainX_flattened,fmt='%f', delimiter=',')
        np.savetxt(os.path.join(script_dir, 'trainY_descaled.csv'), trainY_flattened,fmt='%f', delimiter=',')

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

        print("Predicted subscriber count -- ", subscriber_count_predicted_value, len(subscriber_count_predicted_value))

        original = self.scaler.inverse_transform(np.reshape(np.repeat(self.testY, 14, axis=-1), (len(self.testY), 14)))[:, 6]
        print("\nOriginal Values -- ", original, len(original))

    def split(self):
        date_time = pd.to_datetime(self.df.pop('timestamp'), format='%Y-%m-%d %H:%M:%S')
        column_indices = {name: i for i, name in enumerate(self.df.columns)}
        self.scaled_df = pd.DataFrame(self.scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)

        n = len(self.scaled_df)
        self.train_df = self.scaled_df[0:int(n*0.7)]
        self.val_df = self.scaled_df[int(n*0.7):int(n*0.9)]
        self.test_df = self.scaled_df[int(n*0.9):]

        #self.single_step_models(column_indices)
        self.multi_step_models(self)

    def single_step_models(self, column_indices):

        num_features = self.scaled_df.shape[1]

        single_step_window = WindowGenerator(
                input_width=1, label_width=1, shift=1,
                train_df=self.train_df, val_df=self.val_df, test_df=self.test_df, label_columns=['phoenix_memory_used_cm_sessionP_smf'])
        
        wide_window = WindowGenerator(
                input_width=24, label_width=24, shift=1,
                train_df=self.train_df, val_df=self.val_df, test_df=self.test_df, label_columns=['phoenix_memory_used_cm_sessionP_smf'])
        
        baseline_model=Models(column_indices, wide_window)
        baseline_model.create_baseline_model()

        linear_model = Models(column_indices, wide_window)
        densed_model = Models(column_indices, wide_window)

        linear_model_val_performance, performance = linear_model.performance_evaluation('linear')
        print(linear_model_val_performance, performance)

        densed_model_val_performance, performance = densed_model.performance_evaluation('densed')
        print(densed_model_val_performance, performance)
        
        
        # for example_inputs, example_labels in single_step_window.train.take(1):
        #     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        #     print(f'Labels shape (batch, time, features): {example_labels.shape}')

        # w1 = WindowGenerator(input_width=24, label_width=1, 
        #             shift=24, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df, label_columns=['phoenix_memory_used_cm_sessionP_smf'])
        
        # w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
        #              train_df=self.train_df, val_df=self.val_df, test_df=self.test_df, label_columns=['phoenix_memory_used_cm_sessionP_smf'])
        
        # Stack three slices, the length of the total window.
        # example_window = tf.stack([np.array(self.train_df[:w2.total_window_size]),
        #                    np.array(self.train_df[100:100+w2.total_window_size]),
        #                    np.array(self.train_df[200:200+w2.total_window_size])])
        
        #example_inputs, example_labels = w2.split_window(example_window)

        #w2.plot()

        # print('All shapes are: (batch, time, features)')
        # print(f'Window shape: {example_window.shape}')
        # print(f'Inputs shape: {example_inputs.shape}')
        # print(f'Labels shape: {example_labels.shape}')

        # w1 = window_1.__repr__
        # print(w1)
            
    def multi_step_models(self, column_indices):

        CONV_WIDTH = 3
        conv_window = WindowGenerator(
            input_width=CONV_WIDTH,
            label_width=1,
            shift=1, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
            label_columns=['phoenix_memory_used_cm_sessionP_smf'])

        multi_step_model = Models(column_indices, conv_window)
        multi_step_model_val_performance, multi_step_model_performance = multi_step_model.performance_evaluation('multi_step_densed', conv_window)

        conv_window.plot()
        plt.title("Given 3 hours of inputs, predict 1 hour into the future.")


