from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(5, 13)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))
    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

def retrain_model(trainX, trainY, testX, testY):
    grid_model = KerasRegressor(build_fn=build_model, verbose=1, validation_data=(testX, testY))
    parameters = {'batch_size': [16, 20],
                  'epochs': [8, 10],
                  'optimizer': ['adam', 'Adadelta']}

    grid_search = GridSearchCV(estimator=grid_model, param_grid=parameters, cv=2)
    grid_search = grid_search.fit(trainX, trainY)
    print("Grid search completed.")
    my_model = grid_search.best_estimator_.model
    return my_model
