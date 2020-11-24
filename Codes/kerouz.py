from keras.layers import *
from keras.models import Sequential
from keras.preprocessing import sequence

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

def toknz(pred_corpus, l1,tokenizer):
    seqed_corpus = tokenizer.texts_to_sequences(pred_corpus)  # 将数据集序列化，就是把句中每一个单词编号
    X_out = sequence.pad_sequences(seqed_corpus, maxlen=l1, padding='post')  # 填充与截断序列，就是使得每句话对应的序列长度都是'maxlen'
    return X_out,tokenizer