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
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import math

from tnet.nn import Module, InputInfo

__all__ = [
    "Dropout",
]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

class Dropout(Module):
    """docstring for Dropout."""
    def __init__(self, p):
        self._p = p
        super(Dropout, self).__init__()


    def _declare(self):
        pass


    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)

        if not self.is_in_training:
            return inp


        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)

        mask = rng.binomial(n=1, p=1 - self._p, size=inp.shape)
        y = inp * T.cast(mask, config.floatX)

        return y
