import itertools
import matplotlib.pyplot as plt
import numpy as np
# from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from imutils import paths

n_hidden = 36
n_classes = 5

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
# y_test_path = DATASET_PATH + "Y_test.txt"
y_test_path = DATASET_PATH + "Y_test2.txt"

n_steps = 24  # 32 frames per series

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
X_test = load_X(x_test_path)
y_test = load_y(y_test_path)

filePaths = list(paths.list_files('logs'))
# for filePath in filePaths:
#model34
model.load_weights('logs/model.48-0.78.h5')
# model2 = load_model('models/mymodel_ext.h5')
pred = model.predict(X_test).argmax(axis=-1)
array = confusion_matrix(y_test,pred)
# array = list(array)
    # acc = (array[0][0]+array[1][1]+array[2][2]+array[3][3]+array[4][4])/5
    # print(filePath+" "+str(acc))

df_cm = pd.DataFrame(array, index=["Eat", "Sit", "Sleep", "Stand", "Walk"], columns=["Eat", "Sit", "Sleep", "Stand", "Walk"])
# plt.figure(figsize=(10,7))
df_norm_col = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

sn.set(font_scale=1.4) # for label size
sn.heatmap(df_norm_col, annot=True, annot_kws={"size": 16}, cmap="Blues") # font size
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('ghhgh.jpg')
plt.show()
