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
        self.config = config
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.scaled_df = None
        self.input_width = None
        self.label_width = None
        self.shift = None
        self.label_columns = None
        self.val_mae_val = []
        self.val_mae_test=[]
        self.data = None

    def split(self):
        date_time = pd.to_datetime(self.df.pop('timestamp'), format='%Y-%m-%d %H:%M:%S')
        column_indices = {name: i for i, name in enumerate(self.df.columns)}
        self.scaled_df = pd.DataFrame(self.scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)

        n = len(self.scaled_df)
        self.train_df = self.scaled_df[0:int(n*0.7)]
        self.val_df = self.scaled_df[int(n*0.7):int(n*0.9)]
        self.test_df = self.scaled_df[int(n*0.9):]
        # self.train_df.to_csv('train_data.csv', index=False)
        # self.val_df.to_csv('val_data.csv', index = False)

        self.single_step_models(column_indices)
        self.multi_step_models(self)


    def single_step_models(self, column_indices):

        num_features = self.scaled_df.shape[1]

        single_step_window = WindowGenerator(
                input_width=1, label_width=1, shift=1,
                train_df=self.train_df, val_df=self.val_df, test_df=self.test_df, label_columns=['phoenix_memory_used_cm_sessionP_smf'])
        
        wide_window = WindowGenerator(
                input_width=24, label_width=24, shift=24,
                train_df=self.train_df, val_df=self.val_df, test_df=self.test_df, label_columns=['phoenix_memory_used_cm_sessionP_smf'])
        
        # baseline_model=Models(column_indices, wide_window)
        # baseline_model.create_baseline_model()

        linear_model = Models(column_indices, wide_window)
        densed_model = Models(column_indices, wide_window)

        linear_model_val_performance, linear_performance = linear_model.performance_evaluation('linear', wide_window)

        densed_model_val_performance, densed_performance = densed_model.performance_evaluation('densed', wide_window)

        self.val_mae_val.extend([linear_model_val_performance, densed_model_val_performance])
        self.val_mae_test.extend([linear_model_val_performance, densed_model_val_performance])
            
    def multi_step_models(self, column_indices):

        CONV_WIDTH = 3
        conv_window = WindowGenerator(
            input_width=CONV_WIDTH,
            label_width=1,
            shift=1, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
            label_columns=['phoenix_memory_used_cm_sessionP_smf'])

        # multi_step_model = Models(column_indices, conv_window)
        # multi_step_model_val_performance, multi_step_model_performance = multi_step_model.performance_evaluation('multi_step_densed', conv_window)

        # conv_window.plot()
        # plt.title("Given 3 hours of inputs, predict 1 hour into the future.")

        LABEL_WIDTH = 22
        INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
        wide_window = WindowGenerator(
            input_width=INPUT_WIDTH,
            label_width=24,
            shift=24, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
             label_columns=['phoenix_memory_used_cm_sessionP_smf'])
        
        convolutional_model = Models(column_indices, wide_window)
        convo_model_val_performance, convo_step_model_performance = convolutional_model.performance_evaluation('convolutional_model', wide_window)
        #conv_model = convolutional_model.convolutional_model(wide_conv_window)

        lstm_model = Models(column_indices, wide_window)
        lstm = lstm_model.lstm_model()  
        lstm_model_val_performance, lstm_step_model_performance = lstm_model.performance_evaluation('lstm_model', wide_window)


        self.val_mae_val.extend([convo_model_val_performance, lstm_model_val_performance])
        self.val_mae_test.extend([convo_step_model_performance, lstm_step_model_performance])

        print(self.val_mae_val,'\n', self.val_mae_test)
        self.plot_mae_comparison()
    
    def plot_mae_comparison(self):
        val_maes = []
        test_maes = []
        model_names = set()

        for val_mae in self.val_mae_val:
            for model_name, values in val_mae.items():
                val_maes.append(values[1])
                model_names.add(model_name)

        for test_mae in self.val_mae_test:
            for model_name, values in test_mae.items():
                test_maes.append(values[1])
                model_names.add(model_name)

        model_names = list(model_names)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([f'{name}_val' for name in model_names], val_maes, color='blue', label='Validation MAE')
        ax.bar([f'{name}_test' for name in model_names], test_maes, color='orange', label='Test MAE')

        ax.set_xlabel('Models')
        ax.set_ylabel('MAE')
        ax.set_title('Comparison of MAE for Different Models')
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

