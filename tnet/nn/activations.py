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
import math

from tnet.nn import Module


__all__ = [
    "Identity",
    "Tanh",
    "Sigmoid",
    "HardSigmoid",
    "SoftMax",
    "SoftPlus",
    "Threshold",
    "ReLU"
]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

class _Activation(Module):
    def __init__(self, func):
        self._func = func
        super(_Activation, self).__init__()


    def _declare(self):
        pass

    def _compile(self):
        mock_input = np.array(np.random.rand((1)),  config.floatX)
        self.forward(mock_input)

    def _update_output(self, inp):

        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)
        return self._func(inp)


class Identity(_Activation):
    def __init__(self):
        super(Identity, self).__init__(lambda x: x)


class Tanh(_Activation):
    def __init__(self):
        super(Tanh, self).__init__(T.tanh)


class Sigmoid(_Activation):
    def __init__(self):
        super(Sigmoid, self).__init__(T.nnet.sigmoid)

class HardSigmoid(_Activation):
    def __init__(self):
        super(HardSigmoid, self).__init__(T.nnet.hard_sigmoid)

class SoftMax(_Activation):
    def __init__(self):
        super(SoftMax, self).__init__(T.nnet.softmax)

class SoftPlus(_Activation):
    def __init__(self):
        super(SoftPlus, self).__init__( T.nnet.softplus)


class Threshold(_Activation):

    def __init__(self, th=1e-6,v=0):
        self.__threshold = th
        self.__val = v
        super(Threshold, self).__init__(lambda x : T.switch(T.lt(x, th), v, x) )

class ReLU(Threshold):

    def __init__(self):
        super(ReLU, self).__init__(0.0)
