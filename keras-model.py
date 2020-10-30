import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from numpy import array

data = pd.read_csv("Online-shoppers-intention.csv")

data[["Weekend", "Revenue"]] = data[["Weekend", "Revenue"]].values.astype(int)

data["VisitorType"] = np.asarray(
    [1 if val == "Returning_Visitor" else 0 for val in data["VisitorType"].values]
)


def normalization(column):
    data[column] = np.asfarray((data[column]) / float(max(data[column]) * 0.99) + 0.01)


column_list = data.columns.tolist()
column_list.insert(0, column_list[-1])
column_list.pop()
data = data[column_list]

input = data.iloc[:, 1:]
output = data.iloc[:, 0]

input_train, input_test, output_train, output_test = train_test_split(
    input, output, test_size=0.15, random_state=42
)
model = Sequential()
model.add(Dense(17, input_dim=17, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(
    input_train,
    output_train,
    epochs=10,
    batch_size=20,
    validation_data=(input_test, output_test),
)

model.save("keras-model.h5")
