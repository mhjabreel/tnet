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

import time
import tnet
from tnet.base import *
from collections import OrderedDict

theano = tnet.theano
T = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config


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
    def _get_delegators(self):
        pass

    def _initialization(self):

        x_input, y_input = self._get_delegators()
        inputs = x_input + [y_input] if type(x_input) == list else [x_input, y_input]

        p_y_given_x = self._network(x_input)

        cost = self._criterion(p_y_given_x, y_input)

        # get all the trainable variables and compute the gradient of the cost
        # for each p in params, compute d_cost/d_p
        params = []
        g_params = []

        for p in self._network.parameters:
            #if hasattr(p, 'grad'):
            if isinstance(p, tnet.Parameter):

                g = T.grad(cost=cost, wrt=p)
                params.append(p)
                g_params.append(g)



        gsup =  OrderedDict()
        for p, g in zip(params, g_params):
            gsup[p.grad] = g# [(p.grad, g) for p, g in ] # this substitution is used to populate the gradient to layers' model

        self._fwd_bwd_step = theano.function(
            inputs= inputs,
            outputs=[p_y_given_x, cost],
            updates=gsup
        )

        self._optimizer.define_updates(params)

    def train(self, dataset_iterator, **training_config):

        max_epoch = 10 # The default value of maximum number of epochs
        if "max_epoch" in training_config:
            max_epoch = training_config["max_epoch"]

        self.on_start.invoke(self, EventArgs())

        epoch = 0
        while epoch < max_epoch:

            epoch += 1
            start_time = time.time()
            self.on_start_poch.invoke(self, OnStartEpochEventArgs(epoch, start_time))

            for sample in dataset_iterator():

                inputs = sample["input"] + [sample["target"]] if \
                        type(sample["input"]) == list else [sample["input"], sample["target"]]

                prob, v_cost = self._fwd_bwd_step(*inputs)

                self.on_forward.invoke(self, OnForwardEventArgs(epoch, v_cost, prob, sample["target"]))

                self.on_backward.invoke(self, TrainingEventArgs(epoch))

                self._optimizer.update()

                self.on_update.invoke(self, TrainingEventArgs(epoch))

            self.on_end_epoch.invoke(self, OnEndEpochEventArgs(epoch, start_time, time.time()))

        self.on_end.invoke(self, EventArgs())

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def model(self):
        return self._network


class OnlineTrainer(Trainer):
    def __init__(self, network, criterion, optimizer):
        super(OnlineTrainer, self).__init__(network, criterion, optimizer)

    def _get_delegators(self):
        if not self._network.input_info is None:
            inf = self._network.input_info
            try:
                _ = len(inf) # model has multiple inputs ?
                x = []
                k = 0
                for i in inf:
                    xi = tnet.Delegator(i.dtype, shape=i.shape, name='train_input_%d' % k, strict=True)
                    x.append(xi)
                    k += 1
            except:
                try:
                    x = tnet.Delegator(inf.dtype, shape=inf.shape, name='train_input', strict=True)
                except:
                    raise ValueError("Unsupported input info %s" % (inf))
        else:
            x = tnet.Delegator('float32', ndim=1, name='train_input')

        y = tnet.Delegator('int32', ndim=0, name='train_target') # labels, presented as 1D vector of [int] labels

        return x, y


class MinibatchTrainer(Trainer):
    def __init__(self, network, criterion, optimizer):
        super(MinibatchTrainer, self).__init__(network, criterion, optimizer)

    def _get_delegators(self):
        if not self._network.input_info is None:

            inf = self._network.input_info

            try:
                _ = len(inf) # model has multiple inputs ?
                x = []
                k = 0

                for i in inf:
                    xi = tnet.Delegator(i.dtype, shape=[None,] + i.shape, name='train_input_%d' % k, strict=True)
                    x.append(xi)
                    k += 1
            except:
                try:
                    x = tnet.Delegator(inf.dtype, shape=[None,] + inf.shape, name='train_input', strict=True)
                except Exception as e:
                    #print(e)
                    raise ValueError("Unsupported input info %s" % (inf))
        else:
            x = tnet.Delegator('float32', ndim=2, name='train_input')

        y = tnet.Delegator('int32', ndim=1, name='train_target')#T.iscalar('y')  # labels, presented as 1D vector of [int] labels

        return x, y



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

