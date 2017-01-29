'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb


from tnet import nn
from tnet.dataset import Dataset, BatchDataset, ShuffleDataset, DatasetIterator
from tnet.dataset.custom_datasets import mnist
from tnet.optimizers import *
from tnet.optimizers.sgdoptimizer import SGDOptimizer
from tnet import meter
from tnet.utils import as_shared

import six.moves.cPickle as pickle
import gzip
import os

import os
import sys
import timeit

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = nn.Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(nn.LookupTable(max_features, embedding_dims))

model.add(nn.Dropout(0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(nn.TemporalConvolution(embedding_dims, nb_filter, filter_length))

model.add(nn.ReLU())
# we use max pooling:
model.add(nn.Max(0, 2))

# We add a vanilla hidden layer:
model.add(nn.Linear(250, hidden_dims))
model.add(nn.Dropout(0.2))
model.add(nn.ReLU())

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(nn.Linear(hidden_dims, 1))
model.add(nn.Sum(0, 1)) # flatten
model.add(nn.Sigmoid())



def get_iterator(data):
    data = BatchDataset(
        dataset=ShuffleDataset(
            dataset=data
        ),
        batch_size=batch_size
    )
    iterator = DatasetIterator(data)

    return iterator

loss_meter  = meter.AverageValueMeter()
acc_meter  = meter.BinaryAccuracyMeter()

def on_sample_handler(args):
    print(args.sample["target"])



def on_start_poch_handler(args):
    model.training()
    loss_meter.reset()
    acc_meter.reset()


def on_forward_handler(args):

    loss_meter.add(args.criterion_output)
    acc_meter.add(args.network_output, args.target)

    sys.stderr.write('epoch: {}; avg. loss: {:2.4f}; avg. acc: {:2.4f}\r'.format(args.epoch, loss_meter.value[0], acc_meter.value))
    sys.stderr.flush()

def on_end_epoch_handler(args):
    print('epoch: {}; avg. loss: {:2.4f}; avg. acc: {:2.4f}'.format(args.epoch, loss_meter.value[0], acc_meter.value))
    print("elapsed time: %2.2f seconds" % (args.end_time - args.start_time))



class IMDBTDataset(Dataset):
    """docstring for MNISTDataset."""
    def __init__(self, data):
        super(IMDBTDataset, self).__init__()
        self.add_attribute("input", np.ndarray)
        self.add_attribute("target", np.ndarray)
        self._dataset = data

    def _get(self, idx):
        return self._dataset[0][idx], self._dataset[1][idx].astype(np.int32)

    @property
    def size(self):
        return self._dataset[0].shape[0]

iterator = get_iterator(IMDBTDataset([X_train, y_train]))

#iterator.on_sample += on_sample_handler



criterion = nn.BCECriterion()

optimizer = SGDOptimizer()
optimizer.on_forward += on_forward_handler
optimizer.on_start_poch += on_start_poch_handler
optimizer.on_end_epoch += on_end_epoch_handler


optimizer.train(model, criterion, iterator, learning_rate=0.1,  maxepoch=nb_epoch)

model.evaluate()

print("Testing")

acc_meter.reset()
loss_meter.reset()

iterator = get_iterator(IMDBTDataset([X_test, y_test]))

for sample in iterator():
    X = sample["input"]
    y = sample["target"]
    p_y_given_x = model.forward(X)
    loss = criterion.forward(p_y_given_x, y)
    loss_meter.add(loss)
    acc_meter.add(p_y_given_x, y)

    sys.stderr.write('testing; avg. loss: {:2.4f}; avg. acc: {:2.4f}\r'.format(loss_meter.value[0], acc_meter.value))
    sys.stderr.flush()

print('testing; avg. loss: {:2.4f}; avg. acc: {:2.4f}'.format(loss_meter.value[0], acc_meter.value))
