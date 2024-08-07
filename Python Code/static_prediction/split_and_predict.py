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
from .autoregressive_model.feedback import FeedBack
import sys
import tensorflow_transform as tft
from .pca import PCA
#tf.compat.v1.disable_eager_execution()


class Predictor:
    def __init__(self, df, config, dataset_name):
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
        self.mae_val = []
        self.mae_test=[]
        self.data = None
        self.pca_train_data = None
        self.pca_val_data = None
        self.pca_test_data = None
        self.dataset_name = dataset_name

    def split(self):
        date_time = pd.to_datetime(self.df.pop('timestamp'), format='%Y-%m-%d %H:%M:%S')
        column_indices = {name: i for i, name in enumerate(self.df.columns)}
        self.scaled_df = pd.DataFrame(self.scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
        print(self.scaled_df)
        n = len(self.scaled_df)
        self.train_df = self.scaled_df[0:int(n*0.7)]
        self.val_df = self.scaled_df[int(n*0.7):int(n*0.9)]
        self.test_df = self.scaled_df[int(n*0.9):]

        print("Number of rows in test data:", len(self.test_df))
        
        if self.config['convert_to_pca']:
            pca = PCA(2)
            self.train_df, self.val_df, self.test_df = pca.convert_to_pca(self.train_df, self.val_df, self.test_df)

        self.train_df.to_csv('train_data_pca.csv', index=False)
        # self.val_df.to_csv('val_data.csv', index = False)

        self.predict(column_indices)

    def predict(self, column_indices):

        num_features = self.train_df.shape[1]
        #print(pca_val_data)

        # wide_window = WindowGenerator(
        #     input_width=24, label_width=24, shift=1)

        # for example_inputs, example_labels in wide_window.train.take(1):
        #     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        #     print(f'Labels shape (batch, time, features): {example_labels.shape}')

       

        wide_window = WindowGenerator(
                input_width=self.config['window_width']['input_width'], label_width=self.config['window_width']['label_width'], shift=self.config['window_width']['shift'],
                train_df=self.train_df, val_df=self.train_df, test_df=self.test_df, dataset_name=self.dataset_name)#, label_columns=[self.config['label_columns']])#+self.config['component']])
        
        for example_inputs, example_labels in wide_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')
        

        
        # baseline_model=Models(column_indices, wide_window)
        # baseline_model.create_baseline_model()

        # if self.config['models']['linear']:
        #     linear_model = Models(column_indices, wide_window, num_features, self.config)
        #     linear_model_val_performance, linear_performance = linear_model.performance_evaluation('linear', wide_window)
        #     self.mae_val.extend([linear_model_val_performance])
        #     self.mae_test.extend([linear_performance])
        # if self.config['models']['densed']:
        #     densed_model = Models(column_indices, wide_window, num_features, self.config)
        #     densed_model_val_performance, densed_performance = densed_model.performance_evaluation('densed', wide_window)
        #     self.mae_val.extend([densed_model_val_performance])
        #     self.mae_test.extend([ densed_performance])
        # if self.config['models']['convolutional']:
        #     convolutional_model = Models(column_indices, wide_window, num_features, self.config)
        #     convo_model_val_performance, convo_step_model_performance = convolutional_model.performance_evaluation('convolutional_model', wide_window)
        #     self.mae_val.extend([convo_model_val_performance])
        #     self.mae_test.extend([convo_step_model_performance])
        # if self.config['models']['lstm']:
        #     lstm_model = Models(column_indices, wide_window, num_features, self.config)
        #     lstm_model_val_performance, lstm_step_model_performance = lstm_model.performance_evaluation('lstm_model', wide_window)
        #     self.mae_val.extend([lstm_model_val_performance])
        #     self.mae_test.extend([lstm_step_model_performance])
        # if self.config['models']['multi_step_linear_single_shot']:
        #     single_shot_linear_model = Models(column_indices, wide_window, num_features, self.config)
        #     single_shot_linear_model_val_performance, single_shot_linear_model_test_performance = single_shot_linear_model.performance_evaluation('single_shot_linear', wide_window)
        #     self.mae_val.extend([single_shot_linear_model_val_performance])
        #     self.mae_test.extend([single_shot_linear_model_test_performance])
        # if self.config['models']['multi_step_densed_model']:
        #     multi_step_densed_model = Models(column_indices, wide_window, num_features, self.config)
        #     multi_step_densed_model_val_performance, multi_step_densed_model_test_performance = multi_step_densed_model.performance_evaluation('multi_step_densed', wide_window)
        #     #self.mae_val.extend([multi_step_densed_model_val_performance])
        #     self.mae_test.extend([multi_step_densed_model_test_performance])

        # if self.config['models']['multi_step_convolutional_model']:
        #     multi_step_conv_model = Models(column_indices, wide_window, num_features, self.config)
        #     multi_step_conv_model_val_performance, multi_step_conv_model_test_performance = multi_step_conv_model.performance_evaluation('multi_step_conv', wide_window)
        #     #self.mae_val.extend([multi_step_conv_model_val_performance])
        #     self.mae_test.extend([multi_step_conv_model_test_performance])        

        if self.config['models']['multi_step_lstm_model']:
            multi_step_lstm_model = Models(column_indices, wide_window, num_features, self.config)
            multi_step_lstm_model_val_performance, multi_step_lstm_model_test_performance = multi_step_lstm_model.performance_evaluation('multi_step_lstm', wide_window)
            #self.mae_val.extend([multi_step_lstm_model_val_performance])
            self.mae_test.extend([multi_step_lstm_model_test_performance])

        # if self.config['models']['autoregressive_lstm']:
        #     autoregressive_feedback_lstm = Models(column_indices, wide_window, num_features, self.config)
        #     multi_step_lstm_ar_val_performance, multi_step_lstm_ar_test_performance = autoregressive_feedback_lstm.performance_evaluation('autoregressive_lstm', wide_window)
        #     self.mae_val.extend([multi_step_lstm_ar_val_performance])
        #     self.mae_test.extend([multi_step_lstm_ar_test_performance])

        #self.plot_mae_comparison()
    
    def plot_mae_comparison(self):
        val_maes = []
        test_maes = []
        model_names = set()

        for val_mae in self.mae_val:
            for model_name, values in val_mae.items():
                val_maes.append(values[1])
                model_names.add(model_name)

        for test_mae in self.mae_test:
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

