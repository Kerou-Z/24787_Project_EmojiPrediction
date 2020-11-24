
import math
import pandas as pd
import kerouz as kr
import yongyizh as yy
from keras.preprocessing.text import Tokenizer # https://keras-cn.readthedocs.io/en/latest/preprocessing/text/
from keras.preprocessing import sequence
from keras.utils import to_categorical

if __name__=='__main__':

    # 可调参数
    offset=2 # l1=l3=l2+offset. 其中l1:截取长度, l2:平均长度, l3:训练长度
    rm_symbols='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

    # 读数据集和字典
    data=pd.read_csv('Train.csv')
    X_train=data['TEXT'].values
    Y_train=data['Label'].values
    Y_train=to_categorical(Y_train)
    f='glove.6B.50d.txt'

    # 训练集去除特殊符号及stopwords
    X_rm=yy.corpus_pre(X_train)

    # 分词及序列化
    tokenizer = Tokenizer(filters=rm_symbols, split=" ", lower=True) # filters：需要去除的符号 lower：转换成小写
    tokenizer.fit_on_texts(X_rm) # Tokenizer读取去除特殊符号的训练集,结果保存在tokenizer句柄内
    X_seq=tokenizer.texts_to_sequences(X_rm) # 将数据集序列化就是把句中每一个单词编号

    # 填充与截断
    l2 = math.ceil(sum([len(s.split(" ")) for s in X_rm])/len(X_rm)) # l2:平均长度
    X_pd=sequence.pad_sequences(X_seq,maxlen=l2+offset,padding='post') # 填充与截断序列，就是使得每句话对应的序列长度都是'maxlen'
    X_ind=tokenizer.word_index

    # 初始化字典
    lookup_dict=yy.dict_pre(f)

    # 根据字典生成weightMatrix
    W=yy.lookup(X_ind,lookup_dict)

    # 训练
    model=kr.model_training(len(X_ind)+1, W, l2+offset, X_pd, Y_train, embed_dim=50, epochs=5, )
    print(model.predict_classes(X_pd[1:13])) #test on some sentences in the train data set