from nltk.corpus import stopwords # https://www.nltk.org/book/ch02.html 页面搜索stopwords
import numpy as np
def corpus_pre(X):  # numpy.ndarray 仅限train.csv的数据结构
    pred_X=[]
    stopper=set(stopwords.words('english'))
    for tweets in X:
        words=tweets.split(" ")
        str = ""
        for word in words:
            if word[0] != "@" and word not in stopper:
                if word[0] == "#":
                    word = word[1:]
                str += word + " "
        pred_X.append(str)
    return pred_X  # 去除停止词后的 list
def dict_pre(file):  # wordembedding
    pred_dict={}
    f=open(file,"r",encoding='utf-8') # ['love',1 0 3 2 5 4 ]
    for line in f:
        words=line.split() # 默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等  ['love',1 0 3 2 5 4 ...(50个) ：coef]
        word=words[0] # 词
        coefs=np.asarray(words[1:],dtype='float') # 编码
        pred_dict[word]=coefs
    f.close()
    return pred_dict
def tokenize(pred_corpus, l1):
    seqed_corpus = tokenizer.texts_to_sequences(pred_corpus)  # 将数据集序列化，就是把句中每一个单词编号
    X_out = sequence.pad_sequences(seqed_corpus, maxlen=l1, padding='post')  # 填充与截断序列，就是使得每句话对应的序列长度都是'maxlen'
    return X_out
def lookup(vocab, pred_dict,embedding_dim):
    vocab_size = len(vocab) + 1 # 因为fit_on_texts时自动去除了1个最不常见的词
    W = np.zeros((vocab_size, embedding_dim))
    for word, i in vocab.items(): # .items返回可遍历的元组数组
        if word in pred_dict: # word是'glove.6B.50d.txt'中的; raw_embedding是'train.csv'中的
            W[i] = pred_dict[word]
    return W  # numpy.ndarray
