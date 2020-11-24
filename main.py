
def corpus_pre(X): # numpy.ndarray 仅限train.csv的数据结构
    return pred_X # 去除停止词后的 list
def dict_pre(file): # wordembedding
    return pred_dict
def lookup(pred_corpus,pred_dict):
    return W # numpy.ndarray
def model():
    return 


if __name__=='__main__':

    # 可调参数
    offset=2 # maxlen-average 
    max_length = math.ceil(sum([len(s.split(" ")) for s in smoothtweets])/len(smoothtweets)) # 每句话平均长度
    # l1:截取长度
    # l2:平均长度
    # l3:训练长度
    # s.t. l1=l3=l2+offset
    
    # 读数据集
    data=pd.read_csv('Train.csv')
    x_train=data['TEXT'].values

    # 数据集预处理
    pred_corpus=corpus_pre(X_train)

    # Tokenizer
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ", lower=True) # filters：需要去除的符号 lower：转换成小写
    tokenizer.fit_on_texts(pred_corpus) # 更新词库
    seqed_corpus=tokenizer.texts_to_sequences(pred_corpus) # 将数据集序列化，就是把句中每一个单词编号
    X_train=sequence.pad_sequences(seqed_corpus,maxlen=average+maxlen_offset,padding='post') # 填充与截断序列，就是使得每句话对应的序列长度都是'maxlen'
    dict_pre(pred_corpus)
    maxlength
    vocab=tokenizer.word_index