# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import math
import pandas as pd
import kerouz as kr
import yongyizh as yy
import numpy as np
from keras.preprocessing.text import Tokenizer # https://keras-cn.readthedocs.io/en/latest/preprocessing/text/
from keras.utils import to_categorical
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# %%
if __name__=='__main__':

    # 可调参数
    offset=2 # l1=l3=l2+offset. 其中l1:截取长度, l2:平均长度, l3:训练长度
    rm_symbols='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    embedding_dim=300


# %%
# 读数据集和字典
data_train=pd.read_csv('../dataset/Train.csv')
X_train=data_train['TEXT'].values
Y_train=data_train['Label'].values
Y_train=to_categorical(Y_train)

data_test=pd.read_csv('../dataset/Test.csv')
X_test=data_test['TEXT'].values

f='../dataset/glove.6B.300d.txt'

emoji_map = pd.read_csv('../dataset/Mapping.csv')


# %%
# 训练集去除特殊符号及stopwords
X_rm=yy.corpus_pre(X_train)

# 分词
tokenizer = Tokenizer(filters=rm_symbols, split=" ", lower=True) # filters：需要去除的符号 lower：转换成小写
tokenizer.fit_on_texts(X_rm) # Tokenizer读取去除特殊符号的训练集,结果保存在tokenizer句柄内

# 序列化,填充与截断
l2 = math.ceil(sum([len(s.split(" ")) for s in X_rm])/len(X_rm)) # l2:平均长度
l1 = l2+offset #截断长度
X_pd,tokenizer = kr.toknz(X_rm, l1,tokenizer)


# %%
#Dict that allocate an id(integer) to every word
ind_dict=tokenizer.word_index

#Dict that allocate an word vector to every word
lookup_dict=yy.dict_pre(f)

# 根据字典生成weightMatrix
W=yy.lookup(ind_dict,lookup_dict,embedding_dim)


# %%
# 训练
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model=kr.model_training(len(ind_dict)+1, W, l2+offset, X_pd, Y_train, embed_dim=embedding_dim, epochs=100)
print(model.predict_classes(X_pd[1:13])) #test on some sentences in the train data set

# %% [markdown]
# ## Predict on test set

# %%
# Prediction on test set
X_test_rm = yy.corpus_pre(X_test)
X_test_pd,_ = kr.toknz(X_test_rm, l1,tokenizer)
label_test = model.predict_classes(X_test_pd)
for i in range(500, 521, 1):
    print(emoji_map['emoticons'][label_test[i]])
    print(X_test[i])

# %% [markdown]
# ## Predict on user input

# %%

user_str = input("input your sentence:")   
#user_str = "I love you"
X_user = np.array([str(user_str)])
print(X_user[0])


# %%
X_user_rm = yy.corpus_pre(X_user)
X_user_pd,_ = kr.toknz(X_user_rm, l1,tokenizer)
label_user = model.predict_classes(X_user_pd)
print(emoji_map['emoticons'][label_user[0]])
print(X_user[0]) 


# %%



# %%



