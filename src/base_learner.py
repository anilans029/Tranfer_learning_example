import pandas as pd
import numpy as np
import tensorflow as tf
import logging as lg
import os
import keras
from keras.layers import Dense,Flatten,LeakyReLU
from keras.models import Sequential


lg.basicConfig(filename = os.path.join("logs","running.log"),
              format = "[%(asctime)s----%(levelname)s----%(module)s---->] : %(message)s",
              level = lg.INFO,
              filemode="a")

def main():

    ### get the data
    (x_train, y_train),(x_test,y_test)= tf.keras.datasets.mnist.load_data()
    x_train = x_train/255
    x_test = x_test/255
    x_train, x_valid = x_train[5000:],x_train[:5000]
    y_train,y_valid = y_train[5000:],y_train[:5000]

    ### defining models
    Layers = [  Flatten(input_shape = [28,28]),
                Dense(300,kernel_initializer = "he_normal"),
                keras.layers.LeakyReLU(),
                Dense(100, kernel_initializer="he_normal"),
                keras.layers.LeakyReLU(),
                Dense(10, activation = "softmax")
    ]

    model = Sequential(Layers)

    model.compile(loss = "sparse_categorical_crossentropy",optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3),
                  metrics = ["accuracy"])
    model.summary()

    ### train the model
    history = model.fit(x_train,y_train, validation_data = (x_valid,y_valid),epochs = 10, verbose = 1 )
    root_dir = os.getcwd()
    model_path = os.path.join(root_dir,"artifacts","models","model.h5")
    model.save(model_path)



if __name__ == "__main__":
    main()

