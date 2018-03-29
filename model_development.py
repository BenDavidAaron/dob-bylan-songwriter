from dylan_functions import get_lookup_table, tokenize_per_character
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

doc = open('cleaned text.txt','r').read()
looker = get_lookup_table(doc)
data = tokenize_per_character(doc, lookup_table=looker, sequence_length=50)

X = data['x']
y = data['y']

#define a basic model

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.35))
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# checkpoint this cause I'm gonna be here all day
filepath="models/2-layer-dylan-net_epoch-{epoch:02d}-loss-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=50, batch_size=len(X), callbacks=callbacks_list)