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


class AdadeltaOptimizer(Optimizer):

    """ADADELTA implementation for SGD http://arxiv.org/abs/1212.5701
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        learning_rate: float >= 0. Learning rate.
            It is recommended to leave it at the default value.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    """

    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-06):
        super(AdadeltaOptimizer, self).__init__()
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
        step = theano.shared(0, name="step")
        updates = OrderedDict()

        for p in params:

            #define accumulator and delta_accumulator
            a = p.zero_like()
            d_a = p.zero_like()

            # update accumulator
            new_a = self.rho * a + (1 - self.rho) * (p.grad ** 2)
            updates[a] = new_a

            # use the new accumulator and the *old* delta_accumulator
            update = p.grad * T.sqrt(d_a + self.epsilon) / T.sqrt(new_a + self.epsilon)
            new_p = p - lr * update
            updates[p] = new_p

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * (update ** 2)
            updates[d_a] = new_d_a
        updates[step] = step + 1

        return updates
