from nltk.corpus import stopwords # https://www.nltk.org/book/ch02.html search keyword:stopwords
import numpy as np
def corpus_pre(X):  # numpy.ndarray only works for data structure in train.csv
    pred_X=[]
    stopper=set(stopwords.words('english'))
    for tweets in X:
        words=tweets.split(" ")
        str = ""
        for word in words:
            if word[0] != "@": # and word not in stopper:
                if word[0] == "#":
                    word = word[1:]
                str += word + " "
        pred_X.append(str)
    return pred_X  # cut off the list after stopword
def dict_pre(file):  # wordembedding
    pred_dict={}
    f=open(file,"r",encoding='utf-8') # ['love',1 0 3 2 5 4 ]
    for line in f:
        words=line.split() # Default to be all null characters, including spaces, line breaks, tabs and so on ['love',1 0 3 2 5 4 ...(In total of 50) ：coef]
        word=words[0] # word
        coefs=np.asarray(words[1:],dtype='float') # code
        pred_dict[word]=coefs
    f.close()
    return pred_dict

def lookup(vocab, pred_dict,embedding_dim):
    vocab_size = len(vocab) + 1 # in fit_on_texts process, the least common word was removed automatically
    W = np.zeros((vocab_size, embedding_dim))
    for word, i in vocab.items(): # .items return all tuples and arrays which can be traversed
        if word in pred_dict: # word is from 'glove.6B.50d.txt'; raw_embedding is from 'train.csv'
            W[i] = pred_dict[word]
    return W  # numpy.ndarray