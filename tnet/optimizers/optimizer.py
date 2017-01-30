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

import numpy as np
import math
from tnet.base import *

__all__ = [
    "Optimizer",
    "TrainingEventArgs",
    "OnStartEpochEventArgs",
    "OnEndEpochEventArgs",
    "OnForwardEventArgs",
    "OnBackwardEventArgs",
]


class TrainingEventArgs(EventArgs):

    def __init__(self, epoch):
        super(TrainingEventArgs, self).__init__()
        self.epoch = epoch


class OnStartEpochEventArgs(TrainingEventArgs):

    def __init__(self, epoch, start_time):
        super(OnStartEpochEventArgs, self).__init__(epoch)
        self.start_time = start_time

class OnEndEpochEventArgs(TrainingEventArgs):
    def __init__(self, epoch, start_time, end_time):
        super(OnEndEpochEventArgs, self).__init__(epoch)
        self.start_time = start_time
        self.end_time = end_time


class OnForwardEventArgs(TrainingEventArgs):

    def __init__(self, epoch, criterion_output, network_output, target):
        super(OnForwardEventArgs, self).__init__(epoch)
        self.criterion_output = criterion_output
        self.network_output = network_output
        self.target = target

class OnBackwardEventArgs(TrainingEventArgs):

    def __init__(self, epoch, params, params_grade):
        super(OnForwardEventArgs, self).__init__(epoch)
        self.params = params
        self.params_grade = params_grade
        self.target = target

class Optimizer(object):

    def __init__(self, model, criterion):

        self.on_start = EventHook()
        self.on_end = EventHook()

        self.on_forward = EventHook()
        self.on_backward = EventHook()

        self.on_start_poch = EventHook()
        self.on_end_epoch = EventHook()
        self.on_update = EventHook()

        self._model = model
        self._criterion = criterion


    def _initialization(self):
        if not network.input_info is None:
            #raise ValueError("The passed network has no specific input")

            inf = network.input_info
            ndim = len(inf.shape) + 1

            broadcast = (False,) * ndim
            x = T.TensorType(inf.dtype, broadcast)('x')  # data, presented as rasterized images
        else:
            x = T.matrix('x')

        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
                
        raise NotImplementedError

    def train(self, network, criterion, iterator, config):
        self._get_train_fn(network, criterion, learning_rate)

        self.on_start.invoke(EventArgs())
        epoch = 0

        while epoch < maxepoch:

            epoch += 1
            start_time = time.time()
            self.on_start_poch.invoke(OnStartEpochEventArgs(epoch, start_time))

            for sample in iterator():

                avg_cost, prob = self._train_fn(sample["input"], sample["target"])
                self.on_forward.invoke(OnForwardEventArgs(epoch, avg_cost, prob, sample["target"]))
                self.on_backward.invoke(TrainingEventArgs(epoch))
                self._update_fn(sample["input"], sample["target"])
                self.on_update.invoke(TrainingEventArgs(epoch))


            self.on_end_epoch.invoke(OnEndEpochEventArgs(epoch, start_time, time.time()))

        self.on_end.invoke(EventArgs())


    def test(self, network, criterion, dataset_iterator):
        pass
