import pandas as pd
import numpy as np
import nltk
nltk.download('popular')
import emoji
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.preprocessing import sequence
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
import math

def corpus_pre(X):  # numpy.ndarray 仅限train.csv的数据结构
    return pred_X  # 去除停止词后的 list


def dict_pre(file):  # wordembedding
    return pred_dict

def tokenize(pred_corpus, l1):
    seqed_corpus = tokenizer.texts_to_sequences(pred_corpus)  # 将数据集序列化，就是把句中每一个单词编号
    X_out = sequence.pad_sequences(seqed_corpus, maxlen=l1, padding='post')  # 填充与截断序列，就是使得每句话对应的序列长度都是'maxlen'
    return X_out

def lookup(pred_corpus, pred_dict):
    return W  # numpy.ndarray


def model_training(vocab_size, weight_matrix, l3, X_train, Y_train, embed_dim=50, epochs=5, ):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, weights=[weight_matrix], input_length=l3, trainable=True, ))
    model.add(LSTM(128, dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, epochs=epochs, batch_size=128, shuffle=True, validation_split=0.15)
    model.evaluate(X_train, Y_train)
    return model


if __name__ == '__main__':
    # 可调参数
    offset = 2  # maxlen-average
    max_length = math.ceil(sum([len(s.split(" ")) for s in smoothtweets]) / len(smoothtweets))  # 每句话平均长度
    # l1:截取长度
    # l2:平均长度
    # l3:训练长度
    # s.t. l1=l3=l2+offset

    # 读数据集
    data = pd.read_csv('Train.csv')
    x_train = data['TEXT'].values
    y_train=data['Label'].values
    
    data_test = pd.read_csv('Test.csv')
    x_test = data_test['TEXT'].values
    
    emoji_map = pd.read_csv('Mapping.csv')

    # 数据集预处理
    pred_corpus = corpus_pre(X_train)
    Y_train=to_categorical(y_train)
    
    # Tokenizer
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ",
                          lower=True)  # filters：需要去除的符号 lower：转换成小写
    tokenizer.fit_on_texts(pred_corpus)  # 更新词库
    seqed_corpus = tokenizer.texts_to_sequences(pred_corpus)  # 将数据集序列化，就是把句中每一个单词编号
    X_train = sequence.pad_sequences(seqed_corpus, maxlen=average + maxlen_offset,

                                     padding='post')  # 填充与截断序列，就是使得每句话对应的序列长度都是'maxlen'
    dict_pre(pred_corpus)
    maxlength
    vocab = tokenizer.word_index

    #model design and training
    model = model_training(vocab_size, weight_matrix, l3, X_train, Y_train )
    print(model.predict_classes(X_train[1:13])) #test on some sentences in the train data set

    # Prediction on test set
    pred_corpus_test = corpus_pre(x_test)
    X_test= tokenize(pred_corpus_test, 12)
    label_test = model.predict_classes(X_test)
    for i in range(20, 35, 1):
        print(emoji_map['emoticons'][label_test[i]])
        print(pred_corpus_test[i])
