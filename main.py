from dylan_functions import get_lookup_table, tokenize_per_character

doc = open('cleaned text.txt','r').read()
looker = get_lookup_table(doc)
data = tokenize_per_character(doc, encode=False, lookup_table=looker, sequence_length=10)
