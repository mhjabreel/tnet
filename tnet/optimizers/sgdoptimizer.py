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

import numpy as np
import theano
import math

from tnet.optimizers import Optimizer, TrainingEventArgs, OnForwardEventArgs, OnBackwardEventArgs
from tnet.base import EventArgs, EventHook

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

class SGDOptimizer(Optimizer):

    def __init__(self):
        super(SGDOptimizer, self).__init__()

    def _get_train_fn(self, network, criterion, learning_rate):

        x = T.matrix('x')  # data, presented as rasterized images
        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels


        self._p_y_given_x = network(x)
        self._cost = criterion(self._p_y_given_x, y)

        params = network.parameters
        g_params = [T.grad(cost=self._cost, wrt=p) for p in network.parameters]
        updates = [(p, p - learning_rate * g_p) for p, g_p in zip(network.parameters, g_params)]

        self._train_fn = theano.function(
            inputs=[x, y],
            outputs=[self._cost, self._p_y_given_x],
            updates=updates
        )



    def train(self, network, criterion, iterator, learning_rate=0.13, maxepoch=2):
        self._get_train_fn(network, criterion, learning_rate)

        self.on_start.invoke(EventArgs())
        epoch = 0

        while epoch < maxepoch:

            epoch += 1
            self.on_start_poch.invoke(TrainingEventArgs(epoch))

            for sample in iterator():

                avg_cost, prob = self._train_fn(sample["input"], sample["target"])

                #print(minibatch_avg_cost)
                self.on_forward.invoke(OnForwardEventArgs(epoch, avg_cost, prob, sample["target"]))
                self.on_backward.invoke(TrainingEventArgs(epoch))
                self.on_update.invoke(TrainingEventArgs(epoch))


            self.on_end_epoch.invoke(TrainingEventArgs(epoch))

        self.on_end.invoke(EventArgs())
