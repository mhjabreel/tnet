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

from tnet.optimizers import *
from tnet.base import EventArgs, EventHook

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

class SGDOptimizer(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super(SGDOptimizer, self).__init__()
        self._configs = {
            "learning_rate": learning_rate,
        }
        self._momentum = momentum
        self._nesterov = nesterov

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
            # moment
            sz = p.size()
            m_vals = np.zeros(sz, dtype=p.dtype)
            m = tnet.Variable(m_vals)
            v = self._momentum * m - lr * p.grad  # velocity

            updates[m] = v

            if self._nesterov:
                updates[p] = p + self._momentum * v - lr * p.grad
            else:
                updates[p] = p + v

        updates[step] = step + 1

        return updates


    @property
    def learning_rate(self):
        return self._configs["learning_rate"]

    @learning_rate.setter
    def learning_rate(self, lr):
        assert type(lr) == float
        self._configs["learning_rate"] = lr


    @property
    def momentum(self):
        return self._momentum

    @property
    def nesterov(self):
        return self._nesterov