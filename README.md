# Virtual Bob Dylan

This is a recurent neural network to generate novel Bob Dylan lyrics. 
This repository will be used for deigining a model, once implemented, I'll build a webapp to serve the model.

Thanks to [Blazking](https://github.com/Blaziking) for his [Ngram based Lyric Generator](https://github.com/Blaziking/Lyrics-generation-using-Ngrams), which contained a nicely cleaned and preprocessed set of Dylan lyrics. I've now copied it into my own repo, since I am not using any of his modeling or preprocessing codes, but I want to leave credit where credit is due. 

So Far:
* I've tokenized the text
* Encoded the tokens
* Instantiated a model
* Trained a model for one epic on one batch
* Generated text resembling stuttering (`Hey Mr Tambourine Man Sha Sh sh shee shee`)

TODO:
* Rent some gpu time on GCP or AWS
* Tune the model training and hyperparamaters so it can approximate Dylan's Lyricism