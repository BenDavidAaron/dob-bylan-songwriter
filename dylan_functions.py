import numpy as np

def get_lookup_table(document):
	"""Takes in a text document and returns a per character lookup table
	returns as a dictionary
	"""
	unique_chars = sorted(list(set(document)))
	table = {}
	for label, char in enumerate(unique_chars):
		#table[label] = char
		table[char] = label
	return table

def tokenize_per_character(document, encode=False, 
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
	for idx in range(document_length):
		if idx < sequence_length:
			x_str = document[:idx].rjust(sequence_length)
		else:
			x_str = document[idx-sequence_length:idx].rjust(sequence_length, '\n')

		x = list(x_str)
		y = document[idx]

		if encode:
			try:
				x = [lookup_table[char] for char in x]
				y = lookup_table[y]
			except KeyError:
				print(f"there was an unrecognized character in your document,\nA new table has been generated and can be accessed by checking key: \"lookup\" in the output")
				lookup_table = get_lookup_table(document)
				x = [lookup_table[char] for char in x]
				y = lookup_table[y]
		x_data.append(x)
		y_data.append(y)
	return {"y": y_data, "x": x_data}


#Tests below:
doc = open('cleaned text.txt','r').read()

looker = get_lookup_table(doc)

data = tokenize_per_character(doc, encode=True, sequence_length=10)

text_x = data['x']
test_y = data['y']

for x, y in zip(text_x[:10], test_y[:10]):
	print(x, y)