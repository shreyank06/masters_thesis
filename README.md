```
python3 main.py
```
csv files for amf are stored in Python_Code/amf_csv_files folder. Files are generated after restarting ph_init and executing registration commands with combination given in the names of the csv file names.

To collect csv datas, set "get_csv": to true in configuration files and at the same time set "retrain_model":to false.

After the data have been collected, to train the model, set "retrain_model":true and if you don't want to train and do some predictions on test data set "retrain_model":false,  execute the script with the command python3 main.py
(NOTE as of now target is also provided in the test data, tuning of the model is not set, that's why accuracy could look very vague)

If you set "retrain_model":true and execute the script, training session on the console will look like

```
Epoch 1/8
25/25 [==============================] - 4s 30ms/step - loss: 0.0574 - val_loss: 0.0460
Epoch 2/8
25/25 [==============================] - 0s 9ms/step - loss: 0.0057 - val_loss: 0.0101
Epoch 3/8
25/25 [==============================] - 0s 10ms/step - loss: 0.0040 - val_loss: 0.0031
Epoch 4/8
25/25 [==============================] - 0s 12ms/step - loss: 0.0030 - val_loss: 2.0501e-04
Epoch 5/8
25/25 [==============================] - 0s 9ms/step - loss: 0.0028 - val_loss: 7.3547e-04
Epoch 6/8
25/25 [==============================] - 0s 9ms/step - loss: 0.0028 - val_loss: 0.0013
Epoch 7/8
25/25 [==============================] - 0s 9ms/step - loss: 0.0024 - val_loss: 6.9615e-04
Epoch 8/8
25/25 [==============================] - 0s 10ms/step - loss: 0.0024 - val_loss: 1.5297e-04
25/25 [==============================] - 0s 3ms/step - loss: 2.8317e-04
Epoch 1/8
25/25 [==============================] - 4s 37ms/step - loss: 0.0104 - val_loss: 1.0021e-04
Epoch 2/8
25/25 [==============================] - 0s 11ms/step - loss: 9.1010e-04 - val_loss: 0.0028
Epoch 3/8
25/25 [==============================] - 0s 12ms/step - loss: 7.7945e-04 - val_loss: 0.0024
Epoch 4/8
25/25 [==============================] - 0s 10ms/step - loss: 5.7736e-04 - val_loss: 0.0015
Epoch 5/8
```
