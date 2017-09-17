from __future__ import division, print_function, absolute_import

import tensorflow as tf
import os
import math as mt
from pathlib import Path
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tflearn.data_utils import to_categorical, pad_sequences, load_csv
from keras.layers import Embedding
import numpy as np
from layer import *

#Constants
GLOVE_DIR = "/home/abhishek/tensorflow/Projects/glove.6B"
MAX_SEQUENCE_LENGTH = 20
batch_size = 1024
BUCKETS = [(20, 20)]
NUM_LAYERS = 3
STATE_SIZE = 256
embedding_size = 100
num_classes = 2
vocab_size = 95430
num_steps = 20
learning_rate = 1e-3
total_epochs = 10

#prepare data
#Load from csv
question_file = Path("quo.npz")
if question_file.is_file():
    print("Loading data from quo.npz")
    d1 = np.load(question_file)
    question_pair = d1['question_pair']
    label = d1['label']
else:
    question_pair,label = load_csv ('/home/abhishek/tensorflow/Projects/Quora/train.csv/train.csv',\
                                      target_column=-1, columns_to_ignore=[0,1,2,3], has_header=True,\
                                      categorical_labels=False, n_classes=2)
    np.savez('./quo.npz', question_pair=question_pair, label=label)


#get questions vocabulary
data_file = Path("123.npz")
if data_file.is_file():
    print("Loading data from 123.npz")
    d2 = np.load(data_file)
    x = d2['x']
    y = d2['y']
else:
    vocab_size = 0
    vocab_all = []
    sums = [0, 0]
    for i in range(len(question_pair)):
        for q in range(2):
            ## This following line deletes the ? symbol
            question_pair[i][q] = question_pair[i][q].replace("?", "")
            question_pair[i][q] = question_pair[i][q].replace(",", "")
            question_pair[i][q] = question_pair[i][q].replace(".", "")
            question_pair[i][q] = question_pair[i][q].replace("(", "")
            question_pair[i][q] = question_pair[i][q].replace(")", "")
            question_pair[i][q] = question_pair[i][q].replace('"', "")
            question_pair[i][q] = question_pair[i][q].replace("'", "")
            sums[q] = sums[q] + len(question_pair[i][q].split(" "))
            for words in question_pair[i][q].split(" "):
                vocab_all.append(words)
    vocab_set = set(vocab_all)
    vocab_unique = list(vocab_set)
    vocab_size = len(vocab_unique)
    print("The size of the vocabulary is", vocab_size)

    # Tokenize data
    sequences = []
    seq_data = []
    add_end_of_sentence = []
    tokenizer = Tokenizer(nb_words=vocab_size)
    tokenizer.fit_on_texts(vocab_all)
    x = []
    y = []
    for p in range(len(question_pair)):
        x.append(tokenizer.texts_to_sequences([question_pair[p][0]])[0])
        y.append(tokenizer.texts_to_sequences([question_pair[p][1]])[0])
        if p % 10000 == 0:
            print("I am on the question_pair{}".format(p))
    x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH, padding='post').tolist()
    y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH, padding='post').tolist()
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    x = np.asarray(x)
    y = np.asarray(y)
    print('Shape of sentence 1 tensor:', x.shape)
    print('Shape of sentence 2 tensor:', y.shape)
    print(len(x))
    np.savez('./123.npz', x=x, y=y)

######Creation of batches
def batch_iter(x, y, label, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    x_size = len(x)
    y_size = len(y)
    num_batches_per_epoch = int((len(x)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(x_size))
            shuffled_x = x[shuffle_indices]
            shuffled_y = y[shuffle_indices]
            shuffled_label = label[shuffle_indices]
        else:
            shuffled_x = x
            shuffled_y = y
            shuffled_label = label
        for batch_num in range(num_batches_per_epoch-1):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            yield (shuffled_x[start_index:end_index], shuffled_y[start_index:end_index],shuffled_label[start_index:end_index],epoch)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

reset_graph()
#Placeholders

print('Creating placeholders')
sentence_1 = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='sentence_1')
sentence_2 = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='sentence_2')
labels = tf.placeholder(tf.int32, shape=[None], name='labels')

global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name="Global_step")

embedding = tf.get_variable('embedding_matrix', [vocab_size, STATE_SIZE])
#labels = tf.get_variable('labels', [num_classes])
rnn_input1 = tf.nn.embedding_lookup(embedding,sentence_1)
rnn_input2 = tf.nn.embedding_lookup(embedding,sentence_2)
print(rnn_input2.shape)

#Model

cell = tf.nn.rnn_cell.LSTMCell(STATE_SIZE, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell]*NUM_LAYERS, state_is_tuple=True)
init_state = cell.zero_state(batch_size, tf.float32)
rnn_outputs_1, final_state_1 = tf.nn.dynamic_rnn(cell, rnn_input1, initial_state=init_state)
rnn_outputs_2, final_state_2 = tf.nn.dynamic_rnn(cell, rnn_input2, initial_state=init_state)
print(rnn_outputs_1.shape)
print(final_state_1[2][1])

#final_state = final_state_1[2][1] - final_state_2[2][1]

final_state1 = tf.stack([final_state_1[2][1],final_state_2[2][1]], axis = 2)
#final_state1 = tf.transpose(final_state1)

#Final state Manipulation
with tf.variable_scope('softmax'):
    input_features = STATE_SIZE * 2
    final_state1 = tf.reshape(final_state1, shape=[-1, input_features])
    W = tf.get_variable('W', [input_features, num_classes], initializer=tf.random_normal_initializer(stddev=0.01))
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

#rnn_outputs = tf.reshape(final_state, [-1, STATE_SIZE])
y_reshaped = tf.reshape(labels, [-1])

logits = tf.matmul(final_state1, W) + b
print("Logits are ",logits.shape)
print("Labels are ",labels.shape)
total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y_reshaped))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step=global_step)

with tf.name_scope('summaries'):
    tf.summary.scalar("loss",total_loss)
    tf.summary.histogram("histogram loss", total_loss)
    summary_op = tf.summary.merge_all()

current_epoch = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_size = len(question_pair)
#    writer = tf.summary.FileWriter('./graphs', sess.graph)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    initial_step = global_step.eval()
    training_losses = []
    iterator = batch_iter(x,y,label,batch_size,total_epochs,True)
    while total_epochs>current_epoch:
        training_loss = 0
        steps = 0
        while steps<mt.ceil(total_size/batch_size):
        #for i in range(1):
            steps += 1
            try:
                x, y, lab, ep = next(iterator)
            except:
                break
            loss_, _, summary = sess.run([total_loss, train_step, summary_op], feed_dict={sentence_1: x, sentence_2: y, labels: lab})
            writer.add_summary(summary, global_step=steps)
            training_loss += loss_
            current_epoch = ep
            print("Training loss for step {} of epoch {} is {}".format(steps,current_epoch,loss_))
            if steps%100==0:
                saver.save(sess, 'checkpoints/quora', steps)
        print("Average training loss for Epoch ", current_epoch, ":", training_loss/steps)
        #current_epoch += 1
        training_losses.append(training_loss/steps)
    writer.close()
