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
        self.val_mae = []
        self.perform = []

    def split(self):
        date_time = pd.to_datetime(self.df.pop('timestamp'), format='%Y-%m-%d %H:%M:%S')
        column_indices = {name: i for i, name in enumerate(self.df.columns)}
        self.scaled_df = pd.DataFrame(self.scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)

        n = len(self.scaled_df)
        self.train_df = self.scaled_df[0:int(n*0.7)]
        self.val_df = self.scaled_df[int(n*0.7):int(n*0.9)]
        self.test_df = self.scaled_df[int(n*0.9):]
        self.train_df.to_csv('train_data.csv', index=False)

        self.single_step_models(column_indices)
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

        linear_model_val_performance, linear_performance = linear_model.performance_evaluation('linear', wide_window)

        densed_model_val_performance, densed_performance = densed_model.performance_evaluation('densed', wide_window)

        self.val_mae.extend([linear_model_val_performance, densed_model_val_performance])
        self.perform.extend([linear_performance, densed_performance])
            
    def multi_step_models(self, column_indices):

        CONV_WIDTH = 3
        conv_window = WindowGenerator(
            input_width=CONV_WIDTH,
            label_width=1,
            shift=1, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
            label_columns=['phoenix_memory_used_cm_sessionP_smf'])

        multi_step_model = Models(column_indices, conv_window)
        multi_step_model_val_performance, multi_step_model_performance = multi_step_model.performance_evaluation('multi_step_densed', conv_window)

        # conv_window.plot()
        # plt.title("Given 3 hours of inputs, predict 1 hour into the future.")

        LABEL_WIDTH = 22
        INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
        wide_window = WindowGenerator(
            input_width=INPUT_WIDTH,
            label_width=24,
            shift=1, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
             label_columns=['phoenix_memory_used_cm_sessionP_smf'])
        
        convolutional_model = Models(column_indices, wide_window)
        convo_model_val_performance, convo_step_model_performance = convolutional_model.performance_evaluation('convolutional_model', wide_window)
        #conv_model = convolutional_model.convolutional_model(wide_conv_window)

        lstm_model = Models(column_indices, wide_window)
        lstm = lstm_model.lstm_model()  
        lstm_model_val_performance, lstm_step_model_performance = lstm_model.performance_evaluation('lstm_model', wide_window)

        self.val_mae.extend([multi_step_model_val_performance, convo_model_val_performance, lstm_model_val_performance])
        self.perform.extend([multi_step_model_performance, convo_step_model_performance, lstm_step_model_performance])

        #self.compare_models_performance()
    
    def compare_models_performance(self):
        if not self.val_mae or not self.perform:
            print("Error: Performance data not available.")
            return

        models = list(self.perform[0].keys())  # Get model names from the first performance dictionary
        val_maes = {model: [] for model in models}
        test_maes = {model: [] for model in models}

        for performance in self.val_mae:
            for model, mae in performance.items():
                if model not in val_maes:
                    print(f"Warning: Model '{model}' not found in validation MAE dictionary.")
                else:
                    val_maes[model].append(mae)

        for performance in self.perform:
            for model, mae in performance.items():
                if model not in test_maes:
                    print(f"Warning: Model '{model}' not found in test MAE dictionary.")
                else:
                    test_maes[model].append(mae)

        x = models
        width = 0.35

        fig, ax = plt.subplots()
        print("val_maes:", val_maes)
        print("models:", models)

        ax.bar(np.arange(len(models)), [sum(val_maes[model])/len(val_maes[model]) for model in models], width, label='Validation MAE')
        ax.bar(x + width/2, [sum(test_maes[model])/len(test_maes[model]) for model in models], width, label='Test MAE')
        print("val_maes:", val_maes)
        print("models:", models)


        ax.set_xlabel('Model')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_title('Comparison of Model Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
