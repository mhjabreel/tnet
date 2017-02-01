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

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

class SplitList(Module):
    """
    Creates a module that takes a Variable as input and outputs list of variables,
        splitting the variable along the specified dimension. In the diagram below, dimension is equal to 1.

        +----------+         +-----------+
        | input[1] +---------> {member1, |
     +----------+-+          |           |
      | input[2] +----------->  member2, |
    +----------+-+           |           |
    | input[3] +------------->  member3} |
    +----------+             +-----------+
    The optional parameter nInputDims allows to specify the number of dimensions that this module will receive.
    This makes it possible to forward both minibatch and non-minibatch Tensors through the same module.
    """

    def __init__(self, dim, ndim=None):
        self._dim = dim
        self._ndim = ndim
        super(SplitList, self).__init__()


    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):
        inp = self._check_input(inp)

        dim = self._dim + 1 if not self._ndim is None else self._dim
        splits = tnet.split(inp, dim)
        return splits

    def forward(self, input_or_inputs):
        splits = self._update_output(input_or_inputs)
        print(splits)
        self._output = [s.eval() for s in splits]

        return self._output
class JoinList(Module):
    """docstring for JoinList."""
    def __init__(self, dim, ndim=None):
        self._dim = dim
        self._ndim = ndim
        super(JoinList, self).__init__()

    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):
        dim = self._dim + 1 if not self._ndim is None else self._dim
        inp = self._check_input(inp)
        print(type(inp))
        return T.concatenate([inp], dim)
