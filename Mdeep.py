#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[6]:




# In[2]:


import tensorflow as tf
import numpy as np
import sys
from scipy.cluster.hierarchy import dendrogram, linkage
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.utils import shuffle
import shap
import tensorflow as tf


# In[7]:





# In[ ]:


def network_binary(x_input, args):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=128, strides=64, activation='relu', padding='same', input_shape=(x_input.shape[1], 1)))
    model.add(tf.keras.layers.BatchNormalization(name='bn1'))

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=128, strides=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(name='bn2'))

    model.add(tf.keras.layers.Flatten())

    num_features = model.output_shape[1:]
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(name='bn3'))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model


# In[ ]:


def hac(cor):

    def mydist(p1, p2):
        x = int(p1)
        y = int(p2)
        if cor[x, y] > 1.0:
            return cor[x, y] - 1.0
        return 1.0 - cor[x, y]

    x = list(range(cor.shape[0]))
    X = np.array(x)


    linked = linkage(np.reshape(X, (len(X), 1)), metric=mydist, method='single')
    n = len(linked) + 1
    cache = dict()
    for k in range(len(linked)):
        c1, c2 = int(linked[k][0]), int(linked[k][1])
        c1 = [c1] if c1 < n else cache.pop(c1)
        c2 = [c2] if c2 < n else cache.pop(c2)
        cache[n+k] = c1 + c2
    ix = cache[2*len(linked)]


    return ix


# In[ ]:



np.random.seed(42)
tf.random.set_seed(42)



def new_conv1d_layer(input, filters, kernel_size, strides, keep_prob, name, activation, padding):
    with tf.init_scope():
        layer = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding=padding)(input)

        if activation == 'tanh':
            layer = tf.keras.activations.tanh(layer)
        elif activation == 'relu':
            layer = tf.keras.activations.relu(layer)

        layer = tf.keras.layers.Dropout(rate=0)(layer)

        return layer


def new_activation_layer(input, name):
    with tf.init_scope():
        if name == "tanh":
            layer = tf.keras.activations.tanh(input)
        elif name == "relu":
            layer = tf.keras.activations.relu(input)


        return layers



def new_fc_layer(input, num_inputs, num_outputs, keep_prob, name):
    with tf.compat.v1.variable_scope(name):
        weights = tf.random.truncated_normal([num_inputs, num_outputs], stddev=0.1)
        biases = tf.constant(0.1, shape=[num_outputs])

        dropout = tf.nn.dropout(input, rate=0.2)
        layer = tf.matmul(dropout, weights) + biases


        return layer



def network_continous(x_input, keep_prob, args):
    num_filter = args.kernel_size
    window_size = args.window_size
    stride_size =  args.strides

    layer = new_conv1d_layer(input=x_input, filters=num_filter[0], kernel_size=window_size[0], strides=stride_size[0], keep_prob=keep_prob,name ="conv1",activation='tanh', padding='valid')

    layer = new_conv1d_layer(input=layer, filters=num_filter[1], kernel_size=window_size[1], strides=stride_size[1], keep_prob=keep_prob,name ="conv2",activation='tanh',padding='valid')

    layer = new_conv1d_layer(input=layer, filters=num_filter[2], kernel_size=window_size[2], strides=stride_size[2], keep_prob=keep_prob,name ="conv3",activation='tanh',padding='valid')

    num_features = layer.get_shape()[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])


    layer = new_fc_layer(input=layer, num_inputs=num_features,keep_prob=keep_prob, num_outputs=64, name="fc1")

    layer = new_activation_layer(layer, name="relu")

    layer = new_fc_layer(input=layer, num_inputs=64, keep_prob=keep_prob, num_outputs=8, name="fc2")

    layer = new_activation_layer(layer, name="relu")
    layer = new_fc_layer(input=layer, num_inputs=8, keep_prob=keep_prob, num_outputs=1, name="fc3")

    return layer


def network_binary(x_input, keep_prob, args):
    window_size = [128, 128]
    stride_size = [64, 2]
    num_filter = [32, 32]

    layer = new_conv1d_layer(x_input, num_filter[0], window_size[0], stride_size[0], keep_prob, "conv1", 'relu', 'same')
    layer = tf.keras.layers.BatchNormalization(name='bn1')(layer)

    layer = new_conv1d_layer(layer, num_filter[1], window_size[1], stride_size[1], keep_prob, "conv2", 'relu', 'same')
    layer = tf.keras.layers.BatchNormalization(name='bn2')(layer)
    num_features = layer.get_shape()[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    layer = new_fc_layer(input=layer, num_inputs=num_features, keep_prob=keep_prob, num_outputs=64, name="fc1")
    layer = tf.keras.layers.BatchNormalization(name='bn3')(layer)


    layer = new_fc_layer(input=layer, num_inputs=64, keep_prob=keep_prob,num_outputs=2, name="fc3")

    return layer


def test_mdeep(x_test, args):
    C = np.load('./c.npy')
    hac_index = hac(C)

    x_test = x_test[:, hac_index]
    model = tf.keras.models.load_model('./mdeep_trained.h5')


    outputs = model.predict([x_test, np.ones(len(x_test))])
    y_predict_mdeep = np.where(outputs[:,0] > outputs[:, 1] , 0, 1)

    return y_predict_mdeep

def train_test(x_train, y_train, args):
    n_classes = 2
    num_epochs = args["num_epochs"] #500
    batch_size =  args["batch_size"] #32
    learning_rate = args["learning_rate"] #0.0001
    n_features = args["X_train"].shape[1]
    dropout_rate = args["dropout_rate"] # 0.5

    x = tf.keras.Input(shape=(n_features,))
    y = tf.keras.Input(shape=(n_classes,))


    keep_prob = tf.keras.Input(shape=(), dtype=tf.float32)

    x_input = tf.reshape(x, [-1, n_features, 1])

    layer = network_binary(x_input, keep_prob, args)

    model = tf.keras.Model(inputs=[x, keep_prob], outputs=layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    @tf.function
    def train_step(model, optimizer, x, y, keep_prob):
        with tf.GradientTape() as tape:
            logits = model([x, keep_prob])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(num_epochs):
        x_tmp, y_tmp = shuffle(x_train, y_train)
        total_batch = int(np.shape(x_train)[0] / batch_size)

        for i in range(total_batch - 1):
            x_batch, y_true_batch = x_tmp[i * batch_size:i * batch_size + batch_size],                                     y_tmp[i * batch_size:i * batch_size + batch_size]
            loss = train_step(model, optimizer, x_batch, y_true_batch, dropout_rate)

        loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model([x_train, dropout_rate]), labels=y_train))
        train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model([x_train, dropout_rate]), 1), tf.argmax(y_train, 1)), "float"))

        print("Epoch {}, Loss: {:.4f}, Training accuracy: {:.4f}".format(epoch, loss_val, train_accuracy))

        if train_accuracy > 0.99:
            break
    model.save('./mdeep_trained.h5')
    return test_mdeep(args["X_test"].values, args)


# In[ ]:


def train_test_mdeep(args):
    C = np.load('./c.npy')
    print("Hierarchical clustering")
    hac_index = hac(C)
    print("hac_index")
    print(hac_index)
    print("Start training")
    X_train = args["X_train"]
    x_train = X_train.values
    y_train = args["y_train"]
    y_train_values = y_train.values
    y_tr = []
    for l in y_train_values:
        if l == 1:
            y_tr.append([0, 1])
        else:
            y_tr.append([1, 0])
    y_tr = np.array(y_tr, dtype=int)
    x_train = x_train[:, hac_index]

    return train_test(x_train, y_tr, args)





def model_predict_mdeep(xtest):
    model = tf.keras.models.load_model('./mdeep_trained.h5')
    outputs = model.predict([xtest, np.ones(len(xtest))])
    mdeep_y = np.where((outputs[:,0]) >= (outputs[:, 1]) , 0, 1)
    return mdeep_y
