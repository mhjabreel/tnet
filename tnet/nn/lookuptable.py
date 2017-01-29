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
    "LookupTable",
]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config


class LookupTable(Module):

    """
    Turn positive integers (indexes) into dense vectors of fixed size.

    The parameters are the following:
    n_index: The maximum integer index occurring in the input data.
             This parameter should be int > 0.
    n_output: The dimension of the dense embedding. This parameter should be int > 0.
    #padding_value: . Default is 0

    example:

    """
    def __init__(self, n_index, n_output):

        self._n_index = n_index
        self._n_output = n_output
        self._input_info = InputInfo(dtype='int32', shape=[None])
        super(LookupTable, self).__init__()


    def _compile(self, **kwargs):
        self.forward(np.random.randint(0, self._n_index, (1)))


    def _declare(self):

        nin = self._n_index
        nout = self._n_output
        self._W_values = np.array(np.random.uniform(low=-0.05,
                                              high=0.05,
                                              size=(nin, nout)),
                            theano.config.floatX)

        self._W = theano.shared(self._W_values, borrow=True)

    def _update_output(self, inp):

        if not str(inp.dtype).startswith('int'):
            inp = T.cast(inp, 'int32')
        

        return self._W[inp]

    @property
    def parameters(self):
        return [self._W]

    @property
    def parameter_values(self):
        return [self._W_values]

    @parameter_values.setter
    def parameter_values(self, values):
        self._W_values = values
