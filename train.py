import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import SGD
from clr_callback import *
from keras.callbacks import TensorBoard, ModelCheckpoint

# Useful Constants

# Output classes to learn how to classify
LABELS = [
    "Eat",
    "Sit",
    "Sleep",
    "Stand",
    "Walk"
]
DATASET_PATH = "mydata/data_ext/"

x_train_path = DATASET_PATH + "X_train.txt"
x_test_path = DATASET_PATH + "X_test.txt"
y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

n_steps = 24  # 32 frames per series

# def load_X(path):
#     data = pd.read_csv(path, header=None).values
#     blocks = int(len(data) / n_steps)
#     data = np.array(np.split(data, blocks))
#     return data
#
#
# def load_y(path):
#     data = pd.read_csv(path, header=None).values
#     return data - 1
def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)
    X_ = np.array(np.split(X_, blocks))
    return X_

# Load the networks outputs
def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # for 0-based indexing
    return y_ - 1

# Load the data
X_train = load_X(x_train_path)
X_test = load_X(x_test_path)
y_train = load_y(y_train_path)
y_test = load_y(y_test_path)
# Input Data
training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 1197 test series
n_input = len(X_train[0][0])
n_hidden = 36  # Hidden layer num of features
n_classes = 5  # number of action classes

batch_size = 256

y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
y_test_one_hot = to_categorical(y_test, n_classes)

train_size = X_train.shape[0] - X_train.shape[0] % batch_size
test_size = X_test.shape[0] - X_test.shape[0] % batch_size

model = Sequential([
    # relu activation
    Dense(n_hidden, activation='relu', input_shape=(24, 46)
          ),
    BatchNormalization(),
    LSTM(n_hidden, return_sequences=True, unit_forget_bias=1.0, dropout=0.2),
    LSTM(n_hidden, unit_forget_bias=1.0),
    BatchNormalization(),
    Dense(n_classes,
          activation='softmax'
          )
])

# LR range test
clr = CyclicLR(base_lr=0.0001, max_lr=1, step_size=np.ceil(X_train.shape[0] / (batch_size)), mode='triangular')

model.compile(
    optimizer=SGD(),
    metrics=['accuracy'],
    loss='categorical_crossentropy'
)

history = model.fit(
    X_train[:train_size, :, :],
    y_train_one_hot[:train_size, :],
    epochs=1,
    batch_size=batch_size,
    callbacks=[clr]
)
history = clr.history

my_callbacks = [
    ModelCheckpoint(filepath='logs/model.{epoch:02d}-{val_loss:.2f}.h5'),
    TensorBoard(log_dir='./logs'),
]

clr = CyclicLR(base_lr=0.02, max_lr=0.09, step_size=np.ceil(X_train.shape[0] / (batch_size)), mode='triangular')
history = model.fit(
    X_train[:train_size, :, :],
    y_train_one_hot[:train_size, :],
    epochs=50,
    batch_size=batch_size,
    callbacks=my_callbacks,
    validation_data=(X_test[:test_size, :, :], y_test_one_hot[:test_size, :])
)
# Save the model
# model.save('models/mymodel_ext.h5')

N = 50
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")