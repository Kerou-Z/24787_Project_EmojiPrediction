## Goal
The goal of this project is to generate a proper emoji for an input sentence. This project uses the method of LSTM with Glove embeddings.

## Dataset info
This project uses GloVe for obtaining vector representations for words. We use glove.6B.50d, a pre-trained word vectors file containing 400K different vocabularies, with every vocabulary represented by a 50-dimensional word vector.
glove (embedding representations) download link: https://www.kaggle.com/anindya2906/glove6b

This project uses the Twitter Emoji Prediction dataset. The dataset contains a training dataset of 69832 tweets and their labels (range from 0 to 19) and a test dataset of 25920 tweets. Label-Emoji mapping can be found in the Mapping.csv file.
dataset download link: https://www.kaggle.com/hariharasudhanas/twitter-emoji-prediction?select=Test.csv

## Overview of the project method
1. Preprocess every tweet text by deleting stopwords, hashtag words, etc. 
2. Tokenize every tweet to lower cases without characters other than alphabets.

   Allocate an id(integer) to every word shown in the whole tokenized corpus. This is saved in a dictionary called vocab. #ind_dict = {'word', id}
 
   Transform every tweet text to a word index(id) sequence.
3. Use glove.6B.50d.txt file to create a dictionary lookup_dict. #lookup_dict={'word', 50d_word_vector}
4. Generate a weight matrix in which the i_th column is the 50-dimensional word vector of a word whose id is i, using lookup_dict and ind_dict. 
5. Load the weight matrix into the Keras Embedding layer.
    Connect two LSTM layers to the Embedding layer.
    Use a Dense layer with softmax as an activation function for prediction.
    
## Implementation
 Execute main.py 
 
## Next
wor2vec

Naives Bayes or SVM
