import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

directory = "../data_collection/data/"
file_name = "smf_1000_1_set_1.csv"
file_path = os.path.join(directory, file_name)

df = pd.read_csv(file_path)
date_time = pd.to_datetime(df.pop('timestamp'), format='%Y-%m-%d %H:%M:%S')

print(df.head())
print(df.describe().transpose())
