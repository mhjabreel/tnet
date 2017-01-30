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
import tnet
import time
import numpy as np
import math
from tnet.base import *
import theano

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

__all__ = [
    "Optimizer",
    "Trainer",
    "OnlineTrainer",
    "MinibatchTrainer",
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

class Trainer(object):

    def __init__(self, network, criterion, optimizer):

        self._network = network
        self._criterion = criterion
        self._optimizer = optimizer

        self.on_start = EventHook()
        self.on_end = EventHook()

        self.on_forward = EventHook()
        self.on_backward = EventHook()

        self.on_start_poch = EventHook()
        self.on_end_epoch = EventHook()
        self.on_update = EventHook()

        self._initialization()


    """
    An abstract methdo to get the input and target place holders.
    This method shuld be implemented by the extended classes.
    """
    def _get_placeholders(self):
        pass


    def _initialization(self):

        x_input, y_input = self._get_placeholders()

        p_y_given_x = self._network(x_input)

        cost = self._criterion(p_y_given_x, y_input)

        # get all the trainable variables and compute the gradient of the cost
        # for each p in params, compute d_cost/d_p
        params = []
        g_params = []

        for p in self._network.parameters:
            if isinstance(p, tnet.DifferentiableVariable):
                p.grad.zero() # zero grads
                g = T.grad(cost=cost, wrt=p)
                params.append(p)
                g_params.append(g)


        gsup =  [(p.grad, g) for p, g in zip(params, g_params)] # this substitution is used to populate the gradient to layers' model

        self._fwd_bwd_step = theano.function(
            inputs=[x_input, y_input],
            outputs=[p_y_given_x, cost],
            updates=gsup
        )

        self._optimizer.define_updates(params)

    def train(self, dataset_iterator, **training_config):

        max_epoch = 10 # The default value of maximum number of epochs
        if "max_epoch" in training_config:
            max_epoch = training_config["max_epoch"]

        self.on_start.invoke(EventArgs())

        epoch = 0
        while epoch < max_epoch:

            epoch += 1
            start_time = time.time()
            self.on_start_poch.invoke(OnStartEpochEventArgs(epoch, start_time))

            for sample in dataset_iterator():

                prob, v_cost = self._fwd_bwd_step(sample["input"], sample["target"])

                self.on_forward.invoke(OnForwardEventArgs(epoch, v_cost, prob, sample["target"]))

                self.on_backward.invoke(TrainingEventArgs(epoch))

                self._optimizer.update(**training_config)

                self.on_update.invoke(TrainingEventArgs(epoch))


            self.on_end_epoch.invoke(OnEndEpochEventArgs(epoch, start_time, time.time()))

        self.on_end.invoke(EventArgs())



class OnlineTrainer(Trainer):
    def __init__(self, network, criterion, optimizer):
        super(OnlineTrainer, self).__init__(network, criterion, optimizer)

    def _get_placeholders(self):
        if not self._network.input_info is None:
            #raise ValueError("The passed network has no specific input")

            inf = self._network.input_info
            ndim = len(inf.shape)

            broadcast = (False,) * ndim
            x = T.TensorType(inf.dtype, broadcast)('x')  # data, presented as rasterized images
        else:
            x = T.matrix('x')

        y = T.iscalar('y')  # labels, presented as 1D vector of [int] labels

        return x, y

class MinibatchTrainer(Trainer):
    def __init__(self, network, criterion, optimizer):
        super(MinibatchTrainer, self).__init__(network, criterion, optimizer)

    def _get_placeholders(self):
        if not self._network.input_info is None:
            #raise ValueError("The passed network has no specific input")

            inf = self._network.input_info
            ndim = len(inf.shape) + 1

            broadcast = (False,) * ndim
            x = T.TensorType(inf.dtype, broadcast)('x')  # data, presented as rasterized images
        else:
            x = T.matrix('x')

        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

        return x, y

class Optimizer(object):

    def __init__(self):
        self._defaults = {}


    """
    An abstract methdo to get the placeholders of the optimizer's parameters.
    This method shuld be implemented by the extended classes.
    """
    def _get_placeholders(self):
        pass


    """
    A methdo to get the default values of the optimizer's parameters.
    """
    def _get_default(self, key):

        if key not in self._defaults:
            raise ValueError("Unknown parameter % s" % str(key))

        return self._defaults[key]

    """
    An abstract methdo to get the parameters' update function.
    This method shuld be implemented by the extended optimizers like SGD, Adadelta, ..etc.
    """

    def _get_updates(self, params, inputs):
        pass

    def define_updates(self, params):
        place_holders = self._get_placeholders()
        updates = self._get_updates(params, place_holders)

        self._update_fn = theano.function(inputs=place_holders, outputs=[], updates=updates)


    def update(self, **config):

        pnames = [p.name for p in self._get_placeholders()]
        inputs = []

        for p in pnames:
            if p in config:
                v = config[p]
            else:
                v = self._get_default(p)
            inputs.append(v)

        self._update_fn(*inputs)
