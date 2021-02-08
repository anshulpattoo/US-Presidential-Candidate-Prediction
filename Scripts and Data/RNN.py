import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import sklearn
import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

df = pd.read_csv('/Users/anshulpattoo/Desktop/CISC 251/Project/Python scripts/Spreadsheets/modifiedwinners.csv', index_col=0)


from sklearn.model_selection import train_test_split
features = df.iloc[:, :(df.shape[1] - 1)]
targets = df.iloc[:, (df.shape[1] - 1)]

trainX, testX, trainY, testY = train_test_split(features, targets, test_size = 0.25)

#Scale the data.
scaler = StandardScaler()

scaler.fit(trainX)

trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])

model = Sequential()

# Going to another recurrent layer: return the sequence.
model.add(LSTM(128, input_shape = (trainX.shape[1:]), activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(64, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))

opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

#Mean squared error.
model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = opt,
             metrics = ['accuracy'])
model.fit(trainX, trainY, epochs = 50) #, validation_data = (testX, testY))

predictions = model.predict(testX, batch_size = 10, verbose = 0)

rounded_predictions = model.predict_classes(testX, batch_size = 10, verbose = 0)

actLabels = testY.tolist()
predLabels = rounded_predictions.tolist()

#Accuracy, precision, and recall.
accuracy = sklearn.metrics.accuracy_score(actLabels, predLabels,  normalize = True)
precision = sklearn.metrics.precision_score(actLabels, predLabels, average = 'macro')
recall = sklearn.metrics.recall_score(actLabels, predLabels, average = 'micro')

#Confusion matrix.
data = {'yActual': actLabels, 'yPredicted': predLabels}
df = pd.DataFrame(data, columns=['yActual','yPredicted'])
confusionMatrix = pd.crosstab(df['yActual'], df['yPredicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusionMatrix)

print("The prediction accuracy is " + "{:.2f}".format(accuracy * 100) + "% .")
print("The overall precision is " + "{:.2f}".format(precision) + ".") 
print("The recall is " + "{:.2f}".format(recall) + ".")