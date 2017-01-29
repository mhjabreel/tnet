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

import tnet
import time
import numpy as np
import theano
import math

from tnet.optimizers import *
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

        if not network.input_info is None:
            #raise ValueError("The passed network has no specific input")

            inf = network.input_info
            ndim = len(inf.shape) + 1

            broadcast = (False,) * ndim
            x = T.TensorType(inf.dtype, broadcast)('x')  # data, presented as rasterized images
        else:
            x = T.matrix('x')

        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels


        self._p_y_given_x = network(x)

        self._cost = criterion(self._p_y_given_x, y)
        params = []
        g_params = []
        for p in network.parameters:

            if isinstance(p, tnet.DifferentiableVariable):
                g = T.grad(cost=self._cost, wrt=p)
                params.append(p)
                g_params.append(g)


        updates = [(p, p - learning_rate * g) for p, g in zip(params, g_params)]
        bw_updates =  [(p.grad, g) for p, g in zip(params, g_params)]
        self._train_fn = theano.function(
            inputs=[x, y],
            outputs=[self._cost, self._p_y_given_x],
            updates=bw_updates
        )

        #self._backward_fn = theano.function(inputs=[x, y], outputs=[], updates=bw_updates)
        self._update_fn = theano.function(inputs=[x, y], outputs=[], updates=updates)



    def train(self, network, criterion, iterator, learning_rate=0.13, maxepoch=2):
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
