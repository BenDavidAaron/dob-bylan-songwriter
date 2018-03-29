import numpy as np
import sys
from dylan_functions import get_lookup_table, tokenize_per_character
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

doc = open('cleaned text.txt','r').read()
looker = get_lookup_table(doc)
data = tokenize_per_character(doc, lookup_table=looker, sequence_length=10)
vocab_size = len(set(doc))

X = data['x']
y = data['y']

dataX = data['x_raw']
dataY = data['y_raw']

#define a basic model

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.35))
model.add(LSTM(256))
model.add(Dense(y.shape[1], activation='softmax'))

filename = "" #model to load
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([looker[value] for value in pattern]), "\"")

# generate characters
for i in range(1000):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(vocab_size)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = looker[index]
	seq_in = [looker[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")