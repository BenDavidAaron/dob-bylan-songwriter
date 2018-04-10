import numpy as np
from keras.utils import np_utils
from tqdm import tqdm

def get_lookup_table(document):
	"""Takes in a text document and returns a per character lookup table
	returns as a dictionary
	"""
	unique_chars = sorted(list(set(document)))
	table = {}
	for label, char in enumerate(unique_chars):
		table[char] = label
		table[label] = char
	return table

def tokenize_per_character(document, 
	lookup_table={}, sequence_length=10): 
	"""This function yields tuples containing:
	  a character at a given index in the `document` string
	  the `sequence_length preceding characters in the `document` string`
	if `encode` is True, the function will use the dictionary passed in `lookup_table` to integer encode the preceding characters
	if `encode` is False (or a lookup_table is not provided), the function will return the original characters
	"""
	document_length = len(document)
	x_data = []
	y_data = []
	new_lookup_generated = False
	for idx in tqdm(range(document_length)):
		if idx < sequence_length:
			x_str = document[:idx].rjust(sequence_length)
		else:
			x_str = document[idx-sequence_length:idx].rjust(sequence_length, '\n')

		x = list(x_str)
		y = document[idx]

		try:
			x = [lookup_table[char] for char in x]
			y = lookup_table[y]
		except KeyError:
			print("there was an unrecognized character in your document,\nA new table has been generated and can be accessed by checking key: \"lookup\" in the output")
			lookup_table = get_lookup_table(document)
			x = [lookup_table[char] for char in x]
			y = lookup_table[y]
		x_data.append(x)
		y_data.append(y)
		n_points = len(x_data)
	#reshape each training data point
	X = np.reshape(x_data, (n_points, sequence_length, 1))
	#normalize
	X = X / float(len(lookup_table))
	#one-hot encode the outputs
	y = np_utils.to_categorical(y_data)
	return {"y": y, "x": X, 
	"lookup": lookup_table, 
	'x_raw': x_data, 'y_raw': y_data}












#Tests below:
if __name__ == "__main__":
	doc = open('cleaned text.txt','r').read()
	looker = get_lookup_table(doc)
	data = tokenize_per_character(doc, lookup_table=looker, sequence_length=10)
	print(data)
	print(len(data['x']))
