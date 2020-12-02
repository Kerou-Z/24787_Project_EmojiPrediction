
#%%
from gensim.models import Word2Vec
import pandas as pd
from keras.utils import to_categorical
import Embedding as em
from keras.preprocessing.text import Tokenizer
import numpy as np
import Model as mo
import math
#%%
embedding_dim=100
offset=2
# %%
# read dataset and dictionary
data_train=pd.read_csv('../dataset/Train.csv')
X_train=data_train['TEXT'].values
Y_train=data_train['Label'].values
Y_train=to_categorical(Y_train)

# remove special symbols and stopwords from train set
X_rm=em.corpus_pre(X_train)

# segmentation
rm_symbols='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(filters=rm_symbols, split=" ", lower=True) # filters：filter symbols that need to be removed lower：convert to lowercase
tokenizer.fit_on_texts(X_rm) # Tokenizer read train set free of special symbols. Results are stored in tokenize handle.
X_pd,tokenizer = mo.toknz(X_rm, l2+offset,tokenizer)
ind_dict=tokenizer.word_index
# %%
X_seq=[]
for sentence in X_rm:
    words=list(sentence.lower().split())
    X_seq.append(words)
#%%
model = Word2Vec(sentences=X_seq, size=embedding_dim, window=5, min_count=1, workers=4)
model.save("word2vec.model")
# %%
weight=np.load('word2vec.model.wv.vectors.npy')
l2 = math.ceil(sum([len(s.split(" ")) for s in X_rm])/len(X_rm))
model=mo.model_training(len(weight), weight, l2+offset, X_pd, Y_train, embed_dim=embedding_dim, epochs=5)
print(model.predict_classes(X_pd[1:13])) #test on some sentences in the train data set
# %%
