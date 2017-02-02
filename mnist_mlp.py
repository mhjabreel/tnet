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

'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.14% test accuracy after 20 epochs.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tnet import nn
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

import theano

print("Running on: " + theano.config.device)


def get_iterator(data):
    data = BatchDataset(
        dataset=data,
        batch_size=64
    )
    return DatasetIterator(data)

train_dataset, [X_test, y_test] = mnist.get_data()
train_dataset = ShuffleDataset(dataset=train_dataset)
iterator = get_iterator(train_dataset)

loss_meter  = meter.AverageValueMeter()
acc_meter  = meter.AccuracyMeter()


#get_grad_prams_values
model = nn.Sequential() \
    .add(nn.Linear(28 * 28, 512)) \
    .add(nn.ReLU()) \
    .add(nn.Dropout(0.2)) \
    .add(nn.Linear(512, 512)) \
    .add(nn.ReLU()) \
    .add(nn.Dropout(0.2)) \
    .add(nn.Linear(512, 10)) #\
    #.add(nn.SoftMax())

model = nn.Sequential().add(model)
print(model)
def on_sample_handler(args):

    print(args.sample["target"][0])
    args.sample["target"][0] = 0
    print(args.sample["target"][0])


def on_start_poch_handler(args):
    train_dataset.resample()
    model.training()
    loss_meter.reset()
    acc_meter.reset()


def on_forward_handler(args):

    loss_meter.add(args.criterion_output)
    acc_meter.add(args.network_output, args.target)

    sys.stderr.write('epoch: {}; avg. loss: {:2.2f}; avg. acc: {:2.2f}\r'.format(args.epoch, loss_meter.value[0], acc_meter.value))
    sys.stderr.flush()






#iterator.on_sample += on_sample_handler


def eval():
    model.evaluate()


    acc_meter.reset()
    loss_meter.reset()

    p_y_given_x = model.forward(X_test)
    loss = criterion.forward(p_y_given_x, y_test)
    loss_meter.add(loss)
    acc_meter.add(p_y_given_x, y_test)



    print('Test set: Average loss: {:.4f}, Accuracy:{:.0f} % '.format(loss_meter.value[0], acc_meter.value))

def on_end_epoch_handler(args):

    print('epoch: {}; avg. loss: {:2.4f}; A. acc: {:2.4f}'.format(args.epoch, loss_meter.value[0], acc_meter.value))
    print("elapsed time: %d s" % (args.end_time - args.start_time))
    eval()

model.training()

criterion = nn.CrossEntropyCriterion()

optimizer = SGDOptimizer()

print(optimizer)

trainer = MinibatchTrainer(model, criterion, optimizer)
trainer.on_forward += on_forward_handler
trainer.on_start_poch += on_start_poch_handler
trainer.on_end_epoch += on_end_epoch_handler


trainer.train(iterator, learning_rate=0.1, max_epoch=5)
