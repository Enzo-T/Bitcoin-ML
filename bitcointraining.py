# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:24:06 2021

@author: gentl
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as model_selection


def createData():
    fileName = 'bitcoin1.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6), skiprows=1,delimiter=",",dtype='str')
#1a
    data = data[:1313,:].astype(np.float)
    rows,cols = data.shape
    
    x = data[:,:6]
    y = data[:,6]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,
                                         train_size=0.75,test_size=0.25, random_state=101)
    sc = StandardScaler()

    # Standardize the training dataset 
    #(and calculate the mean and standard deviation)
    X_train = sc.fit_transform(x_train)

    #Use this mean and standard deviation 
    #calculated in the training dataset to 
    #standardize the test dataset
    X_test = sc.transform (x_test)


    return X_train, y_train, X_test, y_test


train_data, train_targets, test_data, test_targets = createData()

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape = (train_data.shape[1],) ))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(train_data, train_targets,
                        epochs=100, batch_size=1)

history_dict = history.history
print("history dict.keys():", history_dict.keys())
mae_history = history.history['mae']        

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

predicted_prices = model.predict(test_data)
predicted_prices = predicted_prices.flatten()

plt.plot(mae_history)
plt.xlabel('epoch')
plt.ylabel('mae_history')
plt.show()

print('MAE: ', test_mae_score)
print('MSE: ', test_mse_score)


