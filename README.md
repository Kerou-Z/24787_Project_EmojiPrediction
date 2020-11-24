The goal of this project is to generate a proper emoji for an input sentence. This Project uses method of LSTM with Glove embeddings.

## Data set info
This project uses GloVe for obtaining vector representations for words. We use glove.6B.50d which is a pre-trained word vectors file that contains 400K different vocabularies, with every vocabulary represented by a 50 dimentional word vector.
glove (embedding representations) download link: https://www.kaggle.com/anindya2906/glove6b

This project uses Twitter Emoji Prediction dataset. The dataset contains training dataset of 69832 tweets and their labels (range from 0 to 19), and a test dataset of 25920 tweets. Label-Emoji mapping can be found in Mapping.csv file.
dataset download link: https://www.kaggle.com/hariharasudhanas/twitter-emoji-prediction?select=Test.csv

## Overview of the project method
1. Preporcess every tweet text by deleting stopwords, hashtag words, etc. 
2. Tokenize every tweet to lower cases without character other than alphabets.
   
   Allocate an id(integer) to every word shown in the whole tokenized corpus. This is saved in a dictionary called vocab. #vocab = {'word', id}
    
   Transform every tweet text to a word index(id) sequence.
3. Use glove.6B.50d.txt file to create a dictionary pre_dict. #pre_dict={'word', 50d_word_vector}
4. Generate a weight matrix in which i th column is the 50 dimentional word vector of word whose id is i, using pre_dict and vocab. 
5. Load the weight matrix into the Keras Embedding layer.
    Connect two LSTM layers to the Embedding layer.
    Use a Dense layer with softmax as actvation function for prediction.
    
## Implementation
 Execute main.py 
