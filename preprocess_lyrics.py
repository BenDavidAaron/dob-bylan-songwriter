import unidecode
import re
f_clean = open('cleaned text.txt', 'w')
with open('Dylan, Bob Lyrics, 1962-2001 - preprocessed.txt','r') as f:
	for line in f.readlines():
		line = unidecode.unidecode(line)
		line = line.lower()
		for char in open('delet_chars.txt', 'r').read().split():
			line = line.replace(char, "")
		if line != "\n":
			f_clean.writelines(line)
f_clean.close()
