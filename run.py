import os
import pickle
from pickle import dump
import nltk
import csv
import itertools
import numpy as np

import rnn

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

if os.path.isfile("X") & os.path.isfile("Y"):
    f1 = open("X", "r")
    f2 = open("Y", "r")
    X = pickle.load(f1)
    Y = pickle.load(f2)
    f1.close()
    f2.close()
else:
    with open('data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    X = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    Y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    f1 = open("X", "w+")
    f2 = open("Y", "w+")
    dump(X, f1)
    dump(Y, f2)
    f1.close()
    f2.close()

# print len(X_train), ",", len(X_train[1])
# print len(y_train), ",", len(y_train[1])
# print X_train[1]
# print y_train[1]

X_train = X[:64000]
Y_train = Y[:64000]
X_test = X[64000:]
Y_test = Y[64000:]
print "Data loaded"

model = rnn.RNN(vocabulary_size, 100)
model.init_params()
model.train(X_train, Y_train, 0.1, 100)





