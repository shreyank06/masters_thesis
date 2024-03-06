import tensorflow as tf
from .baseline import Baseline

class Models:
    MAX_EPOCHS = 20

    def __init__(self, column_indices, window_size):
        self.column_indices = column_indices
        self.window_size = window_size

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

    def compile_and_fit(self, model, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=patience,
                                                           mode='min')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(self.window_size.train, epochs=self.MAX_EPOCHS,
                            validation_data=self.window_size.val,
                            callbacks=[early_stopping])
        return history

    def performance_evaluation(self, model_type, conv_window):

        val_performance = {}
        performance = {}

        # if model_type =='linear':
        #     linear = self.linear_model()
        #     history = self.compile_and_fit(linear)
        #     val_performance[model_type] = linear.evaluate(self.window_size.val)
        #     performance[model_type] = linear.evaluate(self.window_size.test, verbose=0)
        #     self.window_size.plot(linear)

        if model_type =='densed':
            densed = self.densed_model()
            history = self.compile_and_fit(densed)
            val_performance[model_type] = densed.evaluate(self.window_size.val)
            performance[model_type] = densed.evaluate(self.window_size.test, verbose=0)
            self.window_size.plot(dataset = 'train', model=densed)
            self.window_size.plot(dataset='test', model=densed)
            self.window_size.plot(dataset = 'example', model=densed)

        elif model_type == 'multi_step_densed':
            multi_step_densed_model = self.multi_step_densed_model(conv_window)
            history = self.compile_and_fit(multi_step_densed_model)
            val_performance[model_type] = multi_step_densed_model.evaluate(conv_window.val)
            performance[model_type] = multi_step_densed_model.evaluate(conv_window.test, verbose=0)
            conv_window.plot(dataset= 'train', model = multi_step_densed_model)
            conv_window.plot(dataset='test', model = multi_step_densed_model)
            

        elif model_type == 'convolutional_model':
            multi_step_densed_model = self.convolutional_model(conv_window)
            history = self.compile_and_fit(multi_step_densed_model)
            val_performance[model_type] = multi_step_densed_model.evaluate(conv_window.val)
            performance[model_type] = multi_step_densed_model.evaluate(conv_window.test, verbose=0)
            conv_window.plot(dataset= 'train', model = multi_step_densed_model)
            conv_window.plot(dataset='test', model = multi_step_densed_model)

        elif model_type == 'lstm_model':
            lstm_model = self.lstm_model()
            history = self.compile_and_fit(lstm_model)
            val_performance['LSTM'] = lstm_model.evaluate(conv_window.val)
            performance['LSTM'] = lstm_model.evaluate(conv_window.test, verbose=0)
            conv_window.plot(dataset = 'train', model = lstm_model)
            conv_window.plot(dataset = 'test', model = lstm_model)

        return val_performance, performance
    
    def multi_step_densed_model(self, conv_window):

        multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
        return multi_step_dense
    
    def convolutional_model(self, wide_conv_window):
        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                kernel_size=(wide_conv_window.input_width,),
                                activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=24),  # Adjust units to match the desired output shape
            tf.keras.layers.Reshape([24, 1])  # Reshape to (batch_size, 24, 1)
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


        

