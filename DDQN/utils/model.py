from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import keras.backend as K
from keras.optimizers import Adam
import tensorflow as tf

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

def model():
    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_dim=5))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=3))
    model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))
    return model

def save(model, episode):
    model.save("models/{}_{}".format("DDQN", episode))

def load():
    custom_objects = {"huber_loss": huber_loss}
    return load_model("models/" + "DDQN_1000", custom_objects=custom_objects)