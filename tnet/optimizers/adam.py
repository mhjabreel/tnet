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
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

class AdamOptimizer(Optimizer):

    """ADAM implementation for SGD
    Adam - A Method for Stochastic Optimization. (http://arxiv.org/abs/1412.6980v8

    # Arguments
    learning_rate: float >= 0. Learning rate.
    beta_1: float, 0 < beta < 1. Generally close to 1.
    beta_2: float, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor.


    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        super(AdamOptimizer, self).__init__()
        self._configs = {
            "learning_rate": learning_rate,
        }
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = to_shared(0, name='iterations')




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

        t = step + 1
        a_t = lr * T.sqrt(1. - self.beta_2 ** t)/(1. - self.beta_1 ** t)

        for p in params:
            value = p.get_value(borrow=True)
            m_prev = p.grad.zero_like()
            v_prev = p.grad.zero_like()

            m_t = self.beta_1 * m_prev + (1.  -self.beta_1) * p.grad
            v_t = self.beta_2 * v_prev + (1. - self.beta_2) * p.grad ** 2
            d_p = a_t * m_t / (T.sqrt(v_t) + self.epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[p] = p - d_p

        updates[step] = t
        return updates


    @property
    def learning_rate(self):
        return self._configs["learning_rate"]


    @learning_rate.setter
    def learning_rate(self, lr):
        assert type(lr) == float
        self._configs["learning_rate"] = lr