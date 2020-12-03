# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-toolsai.jupyter added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\..\..\..\AppData\Local\Temp\887be895-f28b-4405-b684-b5231cc87f32'))
	print(os.getcwd())
except:
	pass
# %%
import math
import pandas as pd
import Model as mo
import Embedding as em
import numpy as np
from keras.preprocessing.text import Tokenizer # https://keras-cn.readthedocs.io/en/latest/preprocessing/text/
from keras.utils import to_categorical
=======
=======
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a
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
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a


# %%
if __name__=='__main__':

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    # adjustable parameter
    offset=2 # l1=l3=l2+offset. namely l1 refers to get length(truncated length), l2 refers to average length, l3 refers to train length
    rm_symbols='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    embedding_dim=50 # Choose from 50, 100 and 300


# %%
# read dataset and dictionary
=======
=======
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a
    # 可调参数
    offset=2 # l1=l3=l2+offset. 其中l1:截取长度, l2:平均长度, l3:训练长度
    rm_symbols='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    embedding_dim=300


# %%
# 读数据集和字典
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a
data_train=pd.read_csv('../dataset/Train.csv')
X_train=data_train['TEXT'].values
Y_train=data_train['Label'].values
Y_train=to_categorical(Y_train)

data_test=pd.read_csv('../dataset/Test.csv')
X_test=data_test['TEXT'].values

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
f='../dataset/glove.6B.'+str(embedding_dim)+'d.txt'
=======
f='../dataset/glove.6B.300d.txt'
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
f='../dataset/glove.6B.300d.txt'
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
f='../dataset/glove.6B.300d.txt'
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a

emoji_map = pd.read_csv('../dataset/Mapping.csv')


# %%
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# remove special symbols and stopwords from train set
X_rm=em.corpus_pre(X_train)

# segmentation
tokenizer = Tokenizer(filters=rm_symbols, split=" ", lower=True) # filters：filter symbols that need to be removed lower：convert to lowercase
tokenizer.fit_on_texts(X_rm) # Tokenizer read train set free of special symbols. Results are stored in tokenize handle.

# vectorize. fill in and truncation
l2 = math.ceil(sum([len(s.split(" ")) for s in X_rm])/len(X_rm)) # l2:average length
l1 = l2+offset #get length (truncated length)
X_pd,tokenizer = mo.toknz(X_rm, l1,tokenizer)
=======
=======
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a
# 训练集去除特殊符号及stopwords
X_rm=yy.corpus_pre(X_train)

# 分词
tokenizer = Tokenizer(filters=rm_symbols, split=" ", lower=True) # filters：需要去除的符号 lower：转换成小写
tokenizer.fit_on_texts(X_rm) # Tokenizer读取去除特殊符号的训练集,结果保存在tokenizer句柄内

# 序列化,填充与截断
l2 = math.ceil(sum([len(s.split(" ")) for s in X_rm])/len(X_rm)) # l2:平均长度
l1 = l2+offset #截断长度
X_pd,tokenizer = kr.toknz(X_rm, l1,tokenizer)
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a


# %%
#Dict that allocate an id(integer) to every word
ind_dict=tokenizer.word_index

#Dict that allocate an word vector to every word
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
lookup_dict=em.dict_pre(f)

#Generate weightMatrix according to dictionary
W=em.lookup(ind_dict,lookup_dict,embedding_dim)


# %%
# train
model=mo.model_training(len(ind_dict)+1, W, l2+offset, X_pd, Y_train, embed_dim=embedding_dim, epochs=5)
=======
=======
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a
lookup_dict=yy.dict_pre(f)

# 根据字典生成weightMatrix
W=yy.lookup(ind_dict,lookup_dict,embedding_dim)


# %%
# 训练
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model=kr.model_training(len(ind_dict)+1, W, l2+offset, X_pd, Y_train, embed_dim=embedding_dim, epochs=100)
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a
print(model.predict_classes(X_pd[1:13])) #test on some sentences in the train data set

# %% [markdown]
# ## Predict on test set

# %%
# Prediction on test set
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
X_test_rm = em.corpus_pre(X_test)
X_test_pd,_ = mo.toknz(X_test_rm, l1,tokenizer)
=======
X_test_rm = yy.corpus_pre(X_test)
X_test_pd,_ = kr.toknz(X_test_rm, l1,tokenizer)
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
X_test_rm = yy.corpus_pre(X_test)
X_test_pd,_ = kr.toknz(X_test_rm, l1,tokenizer)
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
X_test_rm = yy.corpus_pre(X_test)
X_test_pd,_ = kr.toknz(X_test_rm, l1,tokenizer)
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
X_user_rm = em.corpus_pre(X_user)
X_user_pd,_ = mo.toknz(X_user_rm, l1,tokenizer)
=======
X_user_rm = yy.corpus_pre(X_user)
X_user_pd,_ = kr.toknz(X_user_rm, l1,tokenizer)
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
X_user_rm = yy.corpus_pre(X_user)
X_user_pd,_ = kr.toknz(X_user_rm, l1,tokenizer)
>>>>>>> ebf02e25cbb9efdc333698215e08d71e3984152d
=======
X_user_rm = yy.corpus_pre(X_user)
X_user_pd,_ = kr.toknz(X_user_rm, l1,tokenizer)
>>>>>>> 35fde48719941bec5edf0590c285eac23f03fc7a
label_user = model.predict_classes(X_user_pd)
print(emoji_map['emoticons'][label_user[0]])
print(X_user[0]) 


# %%



# %%



