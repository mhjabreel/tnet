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
import numpy as np
import theano
import math
from collections import OrderedDict

from tnet.optimizers import Optimizer
from tnet.base import EventArgs, EventHook

T  = theano.tensor


class RMSpropOptimizer(Optimizer):

    """RMSProp optimizer. rmsprop: Divide the gradient by a running average of its recent magnitude. (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent
    neural networks.
    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.

    """

    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-6):
        super(RMSpropOptimizer, self).__init__()
        self._defaults = {
            "learning_rate": learning_rate,
        }
        self.rho = rho
        self.epsilon = epsilon




    """
    An abstract methdo to get the placeholders of the optimizer's parameters.
    This method shuld be implemented by the extended classes.
    """
    def _get_delegators(self):
        learning_rate = tnet.Delegator('float32', name='learning_rate', ndim=0)#T.scalar(name='learning_rate')
        return [learning_rate]



    """
    An abstract methdo to get the parameters' update function.
    This method shuld be implemented by the extended optimizers like SGD, Adadelta, ..etc.
    """

    def _get_updates(self, params, inputs):

        lr = inputs[0]
        step = tnet.Variable(np.array(0.0).astype(theano.config.floatX))
        updates = OrderedDict()
        for p in params:

            a = p.zero_like()
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * T.square(p.grad)
            updates[a] = new_a
            #update param
            new_p = p - lr * p.grad / (T.sqrt(new_a) + self.epsilon)
            updates[p] = new_p

        updates[step] = step + 1
        return updates
