'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
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

import tnet.cuda as cuda
cuda.device(0)

print("Running on: " + tnet.device)

# set parameters:
max_features = 20000
maxlen = 80
batch_size = 32
embedding_dims = 128
rnn_size = 128
nb_epoch = 15

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
model.add(nn.LSTM(embedding_dims, rnn_size))
model.add(nn.SelectList(-1))
model.add(nn.Select(1, -1))
model.add(nn.Linear(128, 1))
model.add(nn.View(-1))
model.add(nn.Sigmoid())
print(model)



criterion = nn.BCECriterion()



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


def evaluate():
    model.evaluate()

    acc_meter.reset()
    loss_meter.reset()

    p_y_given_x = model.forward(X_test)
    loss = criterion.forward(p_y_given_x, y_test)
    loss_meter.add(loss)
    acc_meter.add(p_y_given_x, y_test)

    print('Test set: Average loss: {:2.4f}, Accuracy:{:.2f} % '.format(loss_meter.value[0], acc_meter.value))


def on_sample_handler(args):
    pass
    #print(args.sample["target"])



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
    print("elapsed time: %d s" % (args.end_time - args.start_time))

    evaluate()



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

iterator.on_sample += on_sample_handler



optimizer = AdamOptimizer()
trainer = MinibatchTrainer(model, criterion, optimizer)
trainer.on_forward += on_forward_handler
trainer.on_start_poch += on_start_poch_handler
trainer.on_end_epoch += on_end_epoch_handler


trainer.train(iterator, max_epoch=nb_epoch)
evaluate()
