import tensorflow as tf
from future import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
x = [2, 3, 7, 8, 9, 10]
y = [11, 12, 13, 14, 15, 16]

print(x)
print(y)
modelr = keras.Sequential([
    keras.layers.Dense(80, activation=tf.nn.relu, input_shape=[1]),
    keras.layers.Dense(80, activation=tf.nn.relu),
    keras.layers.Dense(80, activation=tf.nn.relu),
    keras.layers.Dense(80, activation=tf.nn.relu),
    keras.layers.Dense(80, activation=tf.nn.relu),
    keras.layers.Dense(1)
])


modelr.compile(loss='mean_squared_error',
               optimizer='adam', metrics=['mean_absolute_error',
                                          'mean_squared_error'])


modelr.fit(x, y, epochs=9)
