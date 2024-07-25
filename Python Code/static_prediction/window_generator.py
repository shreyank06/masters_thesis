import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None, dataset_name = None,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.dataset_name = dataset_name
    print(label_columns)
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      print(inputs)
      if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

      # Slicing doesn't preserve static shape information, so set the shapes
      # manually. This way the `tf.data.Datasets` are easier to inspect.
      inputs.set_shape([None, self.input_width, None])
      labels.set_shape([None, self.label_width, None])

      return inputs, labels
  
  def plot(self, dataset='train', model=None, plot_col=None, max_subplots=3):
      # Determine which dataset to use
      if dataset == 'train':
          inputs, labels = next(iter(self.train))
      elif dataset == 'val':
          inputs, labels = next(iter(self.val))
      elif dataset == 'test':
          inputs, labels = next(iter(self.test))
      elif dataset == 'example':
          inputs, labels = self.example
      else:
          raise ValueError("Invalid dataset. Choose 'train', 'val', 'test', or 'example'.")

      # Determine which columns to plot
      if plot_col is not None:
          if plot_col not in self.column_indices:
              raise KeyError(f"'{plot_col}' not found in column indices. Available keys: {list(self.column_indices.keys())}")
          columns_to_plot = [plot_col]
      else:
          # If plot_col is None, plot all columns
          columns_to_plot = list(self.column_indices.keys())

      # Print the columns that will be plotted
      print("Columns to be plotted:", columns_to_plot)

      # Create or use existing directory for saving plots
      predictions_dir = 'predictions'
      if not os.path.exists(predictions_dir):
          os.makedirs(predictions_dir)

      # Create directory for storing results specific to the dataset
      dataset_results_dir = os.path.join(predictions_dir, self.dataset_name)
      os.makedirs(dataset_results_dir, exist_ok=True)

      for col in columns_to_plot:
          plot_col_index = self.column_indices[col]

          plt.figure(figsize=(12, 8))
          plt.ylabel(f'{col} [normed]')
          
          # Plot multiple samples for the current column
          max_n = min(max_subplots, len(inputs))
          for n in range(max_n):
              print(f"Plotting sample {n + 1} for column: {col}")
              plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                      label='Inputs', marker='.', zorder=-10)

              if self.label_columns:
                  label_col_index = self.label_columns_indices.get(col, None)
              else:
                  label_col_index = plot_col_index

              if label_col_index is None:
                  print(f"Label column index for {col} is None, skipping labels and predictions.")
                  continue

              print(f"Plotting labels for sample {n + 1}, column: {col}")
              plt.scatter(self.label_indices, labels[n, :, label_col_index],
                          edgecolors='k', label='Labels', c='#2ca02c', s=64)
              if model is not None:
                  predictions = model(inputs)
                  print(f"Plotting predictions for sample {n + 1}, column: {col}")
                  plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)

          plt.legend()
          plt.xlabel('Time [s]')
          plt.title(f'Column: {col}')
          plt.savefig(f'{dataset_results_dir}/{col}.png')
          plt.close()

      print(f"Plots saved in '{dataset_results_dir}' directory.")


    
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds
  
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

