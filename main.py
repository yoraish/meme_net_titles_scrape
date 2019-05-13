# the main script for training the net on the count_lsts of titles - which are mapped to to scores

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import json
from keras.utils import plot_model


from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten
import numpy

# fix random seed for reproducibility
import keras
from keras.models import Sequential
from keras.layers import Dense
plt.style.use('ggplot')

def plot_history(history):
    print('in history')
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
    plt.show()

def train():
    # load the data
    print("Generating Train Label Data")
    with open("train_db.json") as train_db:
        # create a dict to map ids to class
        title_to_ups = json.load(train_db)
        x_train = []
        y_train = []
        for title, score in title_to_ups.items():
            x_train.append(np.array(eval(title)))
            y_train.append( score)
    
    
    # populate test ids
    print("Generating Test Label Data")
    with open("test_db.json") as test_db:
        # create a dict to map ids to class
        # create a dict to map ids to class
        title_to_ups = json.load(test_db)
        x_test = []
        y_test = []
        for title, score in title_to_ups.items():
            x_test.append(np.array(eval(title)))
            y_test.append( score)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # sanity check
    print('the length of x_train = ', len(x_train))
    print('the length of x_test = ', len(x_test))

    # print(x_train[1000], y_train[1000])



    # ++++++++++++++++++
    num_outputs = 3
    batch_size = 10
    epochs = 16
    word_vector_length = len(x_train[0])
    print('the lengthof the vocab world verctor = ', word_vector_length)
    # ++++++++++++++++++


    # create model






    # '''
    model = Sequential()
    model.add(Dense(word_vector_length, input_dim=word_vector_length, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(word_vector_length, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1280, activation='sigmoid'))
    model.add(Dense(128, activation='softmax'))
    model.add(Dense(num_outputs, activation='softmax'))


    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose = 1,  validation_data=(x_test, y_test))

    # loss, accuracy_train = model.evaluate(x_train, y_train, verbose=False)
    # print("Training Accuracy: {:.4f}".format(accuracy_train))
    # loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    # print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)
    plot_model(model, to_file='model.png')



    # evaluate the model
    # scores = model.evaluate(x_test, y_test)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
    # model_name = 'scrape_0'
    # with open('HIST.txt', 'a+') as spec_file:
    #     spec_file.write("\nName= " + model_name + " | acc train = " + str(accuracy_train) + " | acc test = " + str(scores[1]*100) + "% | loss test = " + str(scores[0]) + " | num. epochs= " + str(epochs) + " | batch size= " + str(batch_size) + "| memes? " + 'title' + '\n')


    # '''

    # Convert class vectors to binary class matrices. This uses 1 hot encoding.

    # y_train_binary = keras.utils.to_categorical(y_train, num_outputs)
    # y_test_binary = keras.utils.to_categorical(y_test, num_outputs)

    # x_train = x_train.reshape(len(x_train), len(x_train[0]),1)
    # x_test = x_test.reshape(len(x_test), len(x_test[0]),1)

    # model = Sequential()
    # model.add(Conv1D(32, (3), input_shape=(word_vector_length,1), activation='relu' ))
    # model.add(Flatten())
    # model.add(Dense(64, activation='softmax'))
    # model.add(Dense(128, activation='sigmoid'))

    # model.add(Dense(num_outputs, activation='softmax'))

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #             optimizer='adam',
    #             metrics=['accuracy'])

    # model.summary()

    # batch_size = 128
    # epochs = 10
    # history = model.fit(x_train, y_train_binary,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         verbose=1,
    #         validation_data=(x_test, y_test_binary))

    # plot_history(history)









if __name__ == "__main__":
    train()