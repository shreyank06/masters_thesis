```
python3 main.py
```
csv files for amf are stored in Python_Code/amf_csv_files folder. Files are generated after restarting ph_init and executing registration commands with combination given in the names of the csv file names.

To collect csv datas, set "get_csv": to true in configuration files and at the same time set "retrain_model":to false.

After the data have been collected, to train the model, set "retrain_model":true and if you don't want to train and do some predictions on test data set "retrain_model":false,  execute the script with the command python3 main.py(NOTE as of now target is also provided in the test data, that's why accuracy will be very very high)
