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
import tnet.cuda as cuda
from tnet.dataset import BatchDataset, ShuffleDataset, DatasetIterator
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

import numpy

numpy.random.seed(1337)  # for reproducibility
cuda.device(0)
print("Running on: " + tnet.device)

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

kw = 3
kh = 3



def get_iterator(data):
    data = BatchDataset(
        dataset=data,
        batch_size=32
    )
    return DatasetIterator(data)

train_dataset, [X_test, y_test] = mnist.get_data()
train_dataset = ShuffleDataset(dataset=train_dataset)
X_test = numpy.reshape(X_test, (X_test.shape[0], 1, 28, 28))

iterator = get_iterator(train_dataset)

loss_meter  = meter.AverageValueMeter()
acc_meter  = meter.AccuracyMeter()

model = nn.Sequential() \
    .add(nn.SpatialConvolution(1, nb_filters, 3, 3)) \
    .add(nn.ReLU()) \
    .add(nn.SpatialConvolution(nb_filters, nb_filters, 3, 3)) \
    .add(nn.ReLU()) \
    .add(nn.SpatialMaxPooling(2, 2)) \
    .add(nn.SpatialDropout(0.25)) \
    .add(nn.Flatten()) \
    .add(nn.Linear(nb_filters * 12 * 12, 128)) \
    .add(nn.ReLU()) \
    .add(nn.Dropout(0.5)) \
    .add(nn.Linear(128, nb_classes)) \
    .add(nn.SoftMax())


def eval():
    model.evaluate()


    acc_meter.reset()
    loss_meter.reset()

    p_y_given_x = model.forward(X_test)
    loss = criterion.forward(p_y_given_x, y_test)
    loss_meter.add(loss)
    acc_meter.add(p_y_given_x, y_test)

    print('Test set: Average loss: {:.4f}, Accuracy:{:.2f} % '.format(loss_meter.value[0], acc_meter.value))


    
criterion = nn.ClassNLLCriterion()

print(model)
def on_sample_handler(args):

    x = args.sample["input"]
    x = numpy.reshape(x, (x.shape[0], 1,  28, 28))
    args.sample["input"] = x


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
    print("elapsed time: %d seconds" % (args.end_time - args.start_time))
    eval()





iterator.on_sample += on_sample_handler

model.training()

criterion = nn.ClassNLLCriterion()

optimizer = SGDOptimizer()

trainer = MinibatchTrainer(model, criterion, optimizer)
trainer.on_forward += on_forward_handler
trainer.on_start_poch += on_start_poch_handler
trainer.on_end_epoch += on_end_epoch_handler

model.training()
trainer.train(iterator, max_epoch=12)

eval()
