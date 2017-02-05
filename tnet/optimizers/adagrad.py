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


class AdagradOptimizer(Optimizer):

    """ADAGRAD implementation for SGD
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        epsilon: float >= 0.

    """

    def __init__(self, learning_rate=0.01, epsilon=1e-06):
        super(AdagradOptimizer, self).__init__()
        self._defaults = {
            "learning_rate": learning_rate,
        }

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

            #define the accumulator
            a = p.zero_like()
            # update accumulator
            new_a = a + p.grad ** 2
            updates[a] = new_a

            # update params
            new_p = p - lr * p.grad / (T.sqrt(new_a) + self.epsilon)
            updates[p] = new_p


        updates[step] = step + 1
        return updates
