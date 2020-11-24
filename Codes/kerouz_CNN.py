from keras.layers import *
from keras.models import Sequential
from keras.preprocessing import sequence

def model_training(vocab_size, weight_matrix, l3, X_train, Y_train, embed_dim=50, epochs=5, ):
#     embedding_layer = Embedding(vocab_size, embed_dim, weights=[weight_matrix], input_length=l3, trainable=False )
#     sequence_input = Input(shape=(l3,), dtype='int32')
#     embedded_sequences = embedding_layer(sequence_input)
#     x = Conv1D(128, 25, activation='relu', padding='same')(embedded_sequences)
#     x = MaxPooling1D(25, padding='same')(x)
#     x = Conv1D(128, 2, activation='relu', padding='same')(x)
#     x = MaxPooling1D(2, padding='same')(x)
#     x = Conv1D(128, 1, activation='relu', padding='same')(x)
#     x = MaxPooling1D(35, padding='same')(x)  # global max pooling
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     preds = Dense(20, activation='softmax')(x)

#     model = model(sequence_input, preds)
#     model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])

#     # happy learning!
#     model.fit(X_train, Y_train, epochs=epochs, batch_size=128, shuffle=True, validation_split=0.15)
    
    
    
    model = Sequential()
    #model.add(InputLayer(shape=(l3,), dtype='int32', name='x_input'))
    model.add(Embedding(vocab_size, embed_dim, weights=[weight_matrix], input_length=l3, trainable=False ))
    model.add(Conv1D(128, 5, activation='relu', padding='same'))
    model.add(MaxPooling1D(5, padding='same'))
    model.add(Conv1D(128, 5, activation='relu', padding='same'))
    model.add(MaxPooling1D(5, padding='same'))
    model.add(Conv1D(128, 5, activation='relu', padding='same'))
    model.add(MaxPooling1D(35, padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

    model.summary()
    model.fit(X_train, Y_train, epochs=epochs, batch_size=128, shuffle=True, validation_split=0.15)
    model.evaluate(X_train, Y_train)
    return model

def toknz(pred_corpus, l1,tokenizer):
    seqed_corpus = tokenizer.texts_to_sequences(pred_corpus)  # 将数据集序列化，就是把句中每一个单词编号
    X_out = sequence.pad_sequences(seqed_corpus, maxlen=l1, padding='post')  # 填充与截断序列，就是使得每句话对应的序列长度都是'maxlen'
    return X_out,tokenizer