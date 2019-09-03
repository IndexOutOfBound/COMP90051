import numpy as np
from keras import Sequential, optimizers
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from sklearn.feature_extraction.text import CountVectorizer
from Pre_Process import X_train, Y_train, X_test, Y_test
from keras import utils
import tensorflow as tf
import matplotlib.pyplot as plt
ngram_size = 3
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

num_classes = np.max(Y_train) + 1
y_train = utils.to_categorical(Y_train, num_classes)
y_test = utils.to_categorical(Y_test, num_classes)

input_dim = X_train.shape[1]
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
print(Y_train.shape)
SVMmodel = Sequential()
SVMmodel.add(Dense(512, activation = 'relu', input_shape=(input_dim,)))
SVMmodel.add(Dropout(0.5))
SVMmodel.add(Dense(num_classes))
SVMmodel.add(Activation('softmax'))
batch_size=4
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
SVMmodel.compile(loss='squared_hinge',
              optimizer=adam,
              metrics=['accuracy'])
SVMmodel.summary()

history = SVMmodel.fit(X_train, y_train,
                    epochs=2,
                     verbose=1,
                    validation_data=(X_test, y_test),
                    batch_size=batch_size)

loss, accuracy = SVMmodel.evaluate(X_train, Y_train, batch_size=batch_size, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = SVMmodel.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)