import argparse
import os
import shutil
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import logging as lg
from src.utils.common import read_yaml, create_directories
import random
import keras
from keras.layers import Dense,Flatten,LeakyReLU
from keras.models import Sequential

STAGE = "STAGE_NAME" ## <<< change stage name 

lg.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=lg.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
lg.basicConfig(filename = os.path.join("logs","running.log"),
              format = "[%(asctime)s----%(levelname)s----%(module)s---->] : %(message)s",
              level = lg.INFO,
              filemode="a")

def odd_or_even(labels):
    for idx,label in enumerate(labels):
        labels[idx] = np.where(label % 2==0,0,1)
    return labels

def main():

    model_path = os.path.join("artifacts","models","model.h5")
    base_model = keras.models.load_model(model_path)

    for layer in base_model.layers[:-1]:
        layer.trainable = False
    base_model.summary()

    (x_train, y_train),(x_test,y_test)= tf.keras.datasets.mnist.load_data()
    x_train = x_train/255
    x_test = x_test/255
    x_train, x_valid = x_train[5000:],x_train[:5000]
    y_train,y_valid = y_train[5000:],y_train[:5000]

    y_test_bin,y_train_bin,y_valid_bin = odd_or_even([y_test,y_train,y_valid])

    base_layers = base_model.layers[:-1]
    new_model = Sequential(base_layers)
    new_model.add(Dense(2, activation="softmax",name = "output_layer"))

    new_model.summary()

    new_model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                      metrics=["accuracy"])

    history = new_model.fit(x_train, y_train_bin, epochs=10,
                            validation_data=(x_valid, y_valid_bin), verbose=2)

    new_model.evaluate(x_test, y_test_bin)

    new_model_path = os.path.join("artifacts", "models", "new_model.h5")
    new_model.save(new_model_path)

if __name__ == '__main__':
   main()