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

    def __init__(self, input_size, output_size, has_bias=True):
        self._input_info = InputInfo(dtype=config.floatX, shape=[input_size])

        self._has_bias = has_bias
        self._input_size = input_size
        self._output_size = output_size

        super(Linear, self).__init__()

    def _declare(self):

        nin = self._input_size
        nout = self._output_size
        stdv = 1. / math.sqrt(nin)

        self._W_values = np.array(np.random.uniform(low=-stdv,
                                              high=stdv,
                                              size=(nin, nout)),
                            theano.config.floatX)

        self._W = theano.shared(self._W_values, borrow=True)


        if self._has_bias:

            self._b_values = np.array(np.random.uniform(low=-stdv,
                                                  high=stdv,
                                                  size=(nout)),
                                theano.config.floatX)

            self._b = theano.shared(self._b_values, borrow=True)



    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)

        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)

        if inp.ndim == 1 or inp.ndim == 2:
            y = T.dot(inp, self._W)
            if self._has_bias:
                y += self._b

            return y
        else:
            raise Exception("input must be vector or matrix")

    @property
    def parameters(self):
        return [self._W, self._b]

    @property
    def parameter_values(self):
        return [self._W_values, self._b_values]

    @parameter_values.setter
    def parameter_values(self, values):
        self._W_values = values[0]
        self._b_values = values[1]
