# Copyright 2017 Mohammed H. Jabreel. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

'''Trains a simple convnet on the MNIST dataset.
Gets to 98.14% test accuracy after 20 epochs.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from tnet import nn
from tnet.dataset import BatchDataset, ShuffleDataset, DatasetIterator
from tnet.dataset import Dataset
from tnet.dataset.custom_datasets import mnist
from tnet.optimizers import *
from tnet.optimizers.sgdoptimizer import SGDOptimizer
from tnet import meter
from tnet.utils import as_shared
from tnet.utils import get_file
from six.moves import cPickle

import gzip
import os

import os
import sys
import timeit

import numpy

numpy.random.seed(1337)  # for reproducibility



def load_batch(fpath, label_key='labels'):

    print(fpath)
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)

    return data, labels

def load_cifar10_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    nb_train_samples = 50000

    x_train = np.zeros((nb_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((nb_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))


    return (x_train, y_train), (x_test, y_test)

batch_size = 32
nb_classes = 10
nb_epoch = 12

nb_filters = 32
# size of pooling area for max pooling


# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = load_cifar10_data()
y_train = y_train.squeeze()
y_test = y_test.squeeze()



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


class CIFAR10TDataset(Dataset):
    """docstring for CIFAR10TDataset."""
    def __init__(self, data):
        super(CIFAR10TDataset, self).__init__()
        self.add_attribute("input", np.ndarray)
        self.add_attribute("target", np.ndarray)
        self._dataset = data

    def _get(self, idx):
        return self._dataset[0][idx], self._dataset[1][idx].astype(np.int32)

    @property
    def size(self):
        return self._dataset[0].shape[0]


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
acc_meter  = meter.AccuracyMeter()
"""
def on_sample_handler(args):

    x = args.sample["input"]
    x = numpy.reshape(x, (x.shape[0], 1,  28, 28))
    args.sample["input"] = x

"""

def on_start_poch_handler(args):
    model.training()
    loss_meter.reset()
    acc_meter.reset()


def on_forward_handler(args):

    loss_meter.add(args.criterion_output)
    acc_meter.add(args.network_output, args.target)

    sys.stderr.write('epoch: {}; avg. loss: {:2.2f}; avg. acc: {:2.2f}\r'.format(args.epoch, loss_meter.value[0], acc_meter.value))
    sys.stderr.flush()

def on_end_epoch_handler(args):
    print('epoch: {}; avg. loss: {:2.2f}; avg. acc: {:2.2f}'.format(args.epoch, loss_meter.value[0], acc_meter.value))
    print("elapsed time: %2.2f seconds" % (args.end_time - args.start_time))
    model.evaluate()
    print("Testing")
    acc_meter.reset()
    loss_meter.reset()
    p_y_given_x = model.forward(X_test)
    loss = criterion.forward(p_y_given_x, y_test)
    loss_meter.add(loss)
    acc_meter.add(p_y_given_x, y_test)
    print('test; avg. loss: {:2.2f}; avg. acc: {:2.2f}'.format(loss_meter.value[0], acc_meter.value))



iterator = get_iterator(CIFAR10TDataset([X_train, y_train]))

#iterator.on_sample += on_sample_handler

model = nn.Sequential() \
    .add(nn.SpatialConvolution(img_channels, 32, 3, 3, 1, 1, 1, 1)) \
    .add(nn.ReLU()) \
    .add(nn.SpatialConvolution(32, 32, 3, 3)) \
    .add(nn.ReLU()) \
    .add(nn.SpatialMaxPooling(2, 2)) \
    .add(nn.Dropout(0.25)) \
    .add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) \
    .add(nn.ReLU()) \
    .add(nn.SpatialConvolution(64, 64, 3, 3)) \
    .add(nn.ReLU()) \
    .add(nn.SpatialMaxPooling(2, 2)) \
    .add(nn.Dropout(0.25)) \
    .add(nn.Flatten()) \
    .add(nn.Linear(2304, 512)) \
    .add(nn.ReLU()) \
    .add(nn.Dropout(0.5)) \
    .add(nn.Linear(512, nb_classes)) \
    .add(nn.SoftMax())



criterion = nn.ClassNLLCriterion()

optimizer = AdadeltaOptimizer()

trainer = MinibatchTrainer(model, criterion, optimizer)
trainer.on_forward += on_forward_handler
trainer.on_start_poch += on_start_poch_handler
trainer.on_end_epoch += on_end_epoch_handler

model.training()
trainer.train(iterator, max_epoch=20)

model.evaluate()

print("Testing")

acc_meter.reset()
loss_meter.reset()

p_y_given_x = model.forward(X_test)
loss = criterion.forward(p_y_given_x, y_test)
loss_meter.add(loss)
acc_meter.add(p_y_given_x, y_test)



print('test; avg. loss: {:2.2f}; avg. acc: {:2.2f}'.format(loss_meter.value[0], acc_meter.value))
