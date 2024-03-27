import tensorflow as tf
from .baseline import Baseline
from .autoregressive_model.feedback import FeedBack
import os
import sys

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import pandas as pd
import matplotlib.pyplot as plt
#tf.config.run_functions_eagerly(True)

class Models:
    MAX_EPOCHS = 50

    def __init__(self, column_indices, window_size, num_features, config):
        self.column_indices = column_indices
        self.window_size = window_size
        self.num_features = num_features
        self.config = config
        self.hp = HyperParameters()
        
        
    def create_baseline_model(self):
        baseline = Baseline(label_index=self.column_indices['phoenix_memory_used_cm_sessionP_smf'])
        baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        val_performance = {}
        performance = {}
        val_performance['Baseline'] = baseline.evaluate(self.window_size.val)
        performance['Baseline'] = baseline.evaluate(self.window_size.test, verbose=0)

        self.window_size.plot(baseline)

    
    def linear_model(self):
        linear = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1)
        ])
        return linear
    
    def densed_model(self):
        dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
        return dense
    
    def multi_step_densed_model(self, wide_window, num_features):

        multi_step_dense_model = tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(wide_window.label_width*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([wide_window.label_width, num_features])
        ])

        return multi_step_dense_model
    
    def convolutional_model(self, wide_conv_window, num_features):
        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                kernel_size=(wide_conv_window.input_width,),
                                activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=24),  # Adjust units to match the desired output shape
            tf.keras.layers.Reshape([wide_conv_window.label_width, num_features])  # Reshape to (batch_size, 24, 1)
        ])
        return conv_model
    
    def lstm_model(self):
        lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
        ])
        return lstm_model

    def multi_step_linear_single_shot(self, wide_window, num_features):
        multi_linear_model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(wide_window.label_width*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([wide_window.label_width, num_features])
    ])
        return multi_linear_model

    def multi_step_convolutional_model(self, wide_window, num_features):
        CONV_WIDTH = 3
        multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(wide_window.label_width*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([wide_window.label_width, num_features])
    ])
        return multi_conv_model
    
    def multi_step_lstm_model(self, wide_window, num_features):
        units = self.hp.Int('units', min_value=16, max_value=128, step=16) 
        multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(units, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(wide_window.label_width*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([wide_window.label_width, num_features])
    ])
        return multi_lstm_model

    def autoregressive_lstm(self, wide_window, num_features):

        autoregressive_feedback_lstm = FeedBack(units=32, out_steps=wide_window.input_width, num_features=num_features)
        return autoregressive_feedback_lstm

    def hyperparameter_tuning(self, model_fn, model_type):
        tuner = RandomSearch(
            model_fn,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='hyperparameter_tuning',
            project_name=f'{model_type}')

        tuner.search(self.window_size.train,
                    validation_data=self.window_size.val,
                    epochs=self.MAX_EPOCHS)
        
        best_hyperparameters = tuner.get_best_hyperparameters()[0]
        best_model = tuner.hypermodel.build(best_hyperparameters)  # Build the model with the best hyperparameters
        
        #best_model = tuner.get_best_models(num_models=1)[0]
        
        best_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        # history = best_model.fit(self.window_size.train, epochs=self.MAX_EPOCHS,
        #                      validation_data=self.window_size.val,
        #                      callbacks=[early_stopping])

        return best_model

    def compile_and_fit(self, model, model_type, patience=10):
        # tuner = RandomSearch(
        #     model,
        #     objective='val_loss',
        #     max_trials=5,
        #     executions_per_trial=1,
        #     directory='hyperparameter_tuning',
        #     project_name=model_type)
        
        # tuner.search(self.window_size.train,
        #              validation_data=self.window_size.val,
        #              epochs=self.MAX_EPOCHS)
        
        # best_hyperparameters = tuner.get_best_hyperparameters()[0]
        # best_model = tuner.hypermodel.build(best_hyperparameters)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=patience,
                                                           mode='min',
                                                           restore_best_weights=True)
        
        hp_learning_rate = self.hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(self.window_size.train, epochs=self.MAX_EPOCHS,
                            validation_data=self.window_size.val,
                            callbacks=[early_stopping])
        
        # Save the trained model with the model type as the filename
        save_dir = os.path.join(os.getcwd(), 'trained_models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save(os.path.join(save_dir, f'{model_type}.h5'))

        return history

    def performance_evaluation(self, model_type, wide_window):
        val_performance = {}
        performance = {}
        model = None
        model_filename = f"{model_type}.h5"
        model_path = os.path.join("trained_models", model_filename)

        if os.path.exists(model_path):
            # If the model file already exists, load it
            model = tf.keras.models.load_model(model_path)
            if self.config['retrain_model']:
                if model.layers[-1].output_shape[1] == self.window_size.train.element_spec[0].shape[1]:
                    history = self.compile_and_fit(model, model_type)
                    model = tf.keras.models.load_model(model_path)
                    self.window_size.plot(dataset='train', model=model)
                    #val_performance[model_type] = model.evaluate(self.window_size.val)
                    performance[model_type] = model.evaluate(self.window_size.test, verbose=0)

                else:
                    print('Model is not trained for desired output shape, creating new model and training again')
                    # Continue with creating a new model
                    model = self.create_model(model_type, wide_window)
                    history = self.compile_and_fit(model, model_type)
                    model.save(model_path)
                    self.performance_evaluation(model_type, wide_window)
            else:
                if model.layers[-1].output_shape[1] == self.window_size.train.element_spec[0].shape[1]:
                    model = tf.keras.models.load_model(model_path)
                    self.window_size.plot(dataset='val', model=model)
                    #val_performance[model_type] = model.evaluate(self.window_size.val)
                    performance[model_type] = model.evaluate(self.window_size.test, verbose=0)
                else:
                    print('Model is not trained for desired output shape, creating new model and training again')
                    # Continue with creating a new model
                    model = self.create_model(model_type, wide_window)
                    history = self.compile_and_fit(model, model_type)
                    model.save(model_path)
                    self.performance_evaluation(model_type, wide_window)
            # history = self.compile_and_fit(model, model_type)
            # history_df = pd.DataFrame(history.history)
            # history_df.loc[:, ['loss', 'val_loss']].plot()
        else:
            # Otherwise, create a new model based on the model type
            print("The model doesnt exist, creating new model")
            model = self.create_model(model_type, wide_window)
            #best_model = self.hyperparameter_tuning(lambda hp: self.create_model(model_type, wide_window, hp=hp), model_type)
            history = self.compile_and_fit(model, model_type)
            history_df = pd.DataFrame(history.history)
            history_df.loc[:, ['loss', 'val_loss']].plot()
            #history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
            plt.show()

            #val_performance[model_type] = model.evaluate(self.window_size.val)
            performance[model_type] = model.evaluate(self.window_size.test, verbose=0)
            self.window_size.plot(dataset='train', model=model)
        #return val_performance, performance
        return 0, performance
    
    def create_model(self, model_type, wide_window, hp = None):
        if model_type == 'linear':
            model = self.linear_model()
        elif model_type == 'densed':
            model = self.densed_model()
        elif model_type == 'convolutional_model':
            model = self.convolutional_model(wide_window, self.num_features)
        elif model_type == 'lstm_model':
            model = self.lstm_model()
        elif model_type == 'single_shot_linear':
            model = self.multi_step_linear_single_shot(wide_window, self.num_features)
        elif model_type == 'multi_step_densed':
            model = self.multi_step_densed_model(wide_window, self.num_features)
        elif model_type == 'multi_step_conv':
            model = self.multi_step_convolutional_model(wide_window, self.num_features)
        elif model_type == 'multi_step_lstm':
            model = self.multi_step_lstm_model(wide_window, self.num_features)
        elif model_type == 'autoregressive_lstm':
            model = self.autoregressive_lstm(wide_window, self.num_features)
        return model
