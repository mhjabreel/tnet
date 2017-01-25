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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tnet import nn
from tnet.dataset import *
from tnet.optimizers import *
from tnet.optimizers.sgdoptimizer import SGDOptimizer
from tnet import meter

import six.moves.cPickle as pickle
import gzip
import os

import os
import sys
import timeit

import numpy


class MNISTDataset(Dataset):
    """docstring for MNISTDataset."""
    def __init__(self, path, mode='train'):
        super(MNISTDataset, self).__init__()
        self.add_attribute("input", np.ndarray)
        self.add_attribute("target", np.ndarray)
        self.__load_data(path, mode)


    def __load_data(self, path='mnist.pkl.gz', mode='train'):

        print('... loading data')

        # Load the dataset

        with gzip.open(path, 'rb') as f:
            if mode == 'train':
                try:
                    data, _, _ = pickle.load(f, encoding='latin1')
                except:
                    data, _, _ = pickle.load(f)
            elif mode == 'dev':

                try:
                    _, data, _ = pickle.load(f, encoding='latin1')
                except:
                    _, data, _ = pickle.load(f)

            elif mode == 'test':

                try:
                    _,_, data = pickle.load(f, encoding='latin1')
                except:
                    _,_, data= pickle.load(f)
        
        self._dataset = data

    def _get(self, idx):

        return self._dataset[0][idx], self._dataset[1][idx].astype(numpy.int32)

    @property
    def size(self):
        return self._dataset[0].shape[0]


def get_iterator(mode='train'):

    mnist_trainset = MNISTDataset('data/mnist.pkl.gz', mode)
    #iterator = DatasetIterator(mnist_trainset)

    data = BatchDataset(
        dataset=ShuffleDataset(
            dataset=mnist_trainset
        ),
        batch_size=128
    )
    return DatasetIterator(data)

loss_meter  = meter.AverageValueMeter()
clerr  = meter.ClassErrorMeter()

def on_sample_handler(args):
    print(args.sample["target"][0])
    args.sample["target"][0] = 0
    print(args.sample["target"][0])


def on_start_poch_handler(args):
    loss_meter.reset()
    clerr.reset()


def on_forward_handler(args):

    loss_meter.add(args.criterion_output)
    clerr.add(args.network_output, args.target)

    #if args.epoch % print_every == 0:
    #
    sys.stderr.write('epoch: {}; avg. loss: {:2.2f}; avg. error: {:2.2f}\r'.format(args.epoch, loss_meter.value[0], clerr.value))
    sys.stderr.flush()

sys.stderr.write("\r\n")
print("")
iterator = get_iterator('train')

#iterator.on_sample += on_sample_handler
model = nn.Sequential()
model.add(nn.Linear(28 * 28, 10))
model.add(nn.SoftMax())
criterion = nn.ClassNLLCriterion()

optimizer = SGDOptimizer()
optimizer.on_forward += on_forward_handler
optimizer.on_start_poch += on_start_poch_handler
optimizer.train(model, criterion, iterator, learning_rate=0.2,  maxepoch=2)


#testing
iterator = get_iterator('test')
clerr.reset()
loss_meter.reset()
for sample in iterator():
    p_y_given_x = model.forward(sample['input'])
    loss = criterion.forward(p_y_given_x, sample['target'])
    loss_meter.add(loss)
    clerr.add(p_y_given_x, sample['target'])

print('test; avg. loss: {:2.2f}; avg. error: {:2.2f}'.format(loss_meter.value[0], clerr.value))
