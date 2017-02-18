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
import tnet
from tnet.nn import Module, InputInfo

__all__ = [
    "Linear",
]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

class Linear(Module):
    """
    Applies a linear transformation to the incoming data, i.e. y = Ax + b.
    The input tensor given in forward(input) must be either a vector (1D tensor) or matrix (2D tensor).
    If the input is a matrix, then each row is assumed to be an input sample of given batch.
    The layer can be used without bias by setting bias = false.

    You can create a layer in the following way:

     module = nn.Linear(10, 5)  -- 10 inputs, 5 outputs
     Usually this would be added to a network of some kind, e.g.:

     net = nn.Sequential()
     net.add(module)
    """
    def __init__(self, input_size, output_size, has_bias=True):
        self._input_info = InputInfo(dtype=config.floatX, shape=[input_size])

        self._has_bias = has_bias
        self._input_size = input_size
        self._output_size = output_size

        super(Linear, self).__init__()

    def _declare(self):

        nin = self._input_size
        nout = self._output_size

        stdv = np.sqrt(6. / (nin + nout))#1. / np.sqrt(nin) #

        w_values = np.array(np.random.uniform(low=-stdv,
                                              high=stdv,
                                              size=(nin, nout)),
                            theano.config.floatX)

        self._W = tnet.Parameter(w_values)
        self._params.append(self._W)

        if self._has_bias:

            # _b_values = np.array(np.random.uniform(low=-stdv,
            #                                       high=stdv,
            #                                       size=( nout)),
            #                     theano.config.floatX)

            _b_values = np.zeros((nout,), dtype=theano.config.floatX)

            self._b = tnet.Parameter(_b_values)
            self._params.append(self._b)

    def _update_output(self, inp):

        inp = super(Linear, self)._update_output(inp)
        if inp.ndim == 1 or inp.ndim == 2:
            y = T.dot(inp, self._W)
            if self._has_bias:
                y += self._b
            return y

        else:
            raise Exception("input must be vector or matrix")

    def __repr__(self):

        return "{}({} -> {})".format(self.__class__.__name__, self._input_size, self._output_size)
