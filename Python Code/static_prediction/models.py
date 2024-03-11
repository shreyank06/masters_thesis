import tensorflow as tf
from .baseline import Baseline

class Models:
    MAX_EPOCHS = 20

    def __init__(self, column_indices, window_size, num_features):
        self.column_indices = column_indices
        self.window_size = window_size
        self.num_features = num_features

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

    def performance_evaluation(self, model_type, wide_window):

        val_performance = {}
        performance = {}

        if model_type =='linear':
            model = self.linear_model()
        if model_type == 'densed':
            model = self.densed_model()
        if model_type == 'convolutional_model':
            model = self.convolutional_model(wide_window)
        if model_type == 'lstm_model':
            model = self.lstm_model()
        if model_type == 'single_shot_linear':
            model = self.multi_step_linear_single_shot(wide_window, self.num_features)
            
        history = self.compile_and_fit(model)
        val_performance[model_type] = model.evaluate(self.window_size.val)
        performance[model_type] = model.evaluate(self.window_size.test, verbose=0)
        self.window_size.plot(dataset = 'train', model = model)
        # self.window_size.plot(dataset = 'example', model = model)
        # self.window_size.plot(dataset = 'test', model = model)

        return val_performance, performance
    
    def multi_step_densed_model(self, wide_window, num_features):

        multi_step_dense_model = tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(wide_window.OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([wide_window.OUT_STEPS, num_features])
        ])

        return multi_step_dense_model
    
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
        multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(wide_window.OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([wide_window.OUT_STEPS, num_features])
    ])


        

