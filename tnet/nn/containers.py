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

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

from tnet.nn import Module

__all__ = [
    "Container",
    "Sequential"
]


class Container(Module):

    def __init__(self):
        self._modules = []
        super(Container, self).__init__()

    def _compile(self):
        pass

    def _declare(self):
        pass

    def add(self, module):
        assert isinstance(module, Module)
        self._modules.append(module)

    @property
    def parameters(self):
        params = []
        for m in self._modules:
            p = m.parameters
            if not p is None:
                params += p
        return params

    @property
    def parameter_values(self):
        param_vals = []
        for m in self._modules:
            v = m.parameter_values
            if not v is None:
                param_vals += v
        return param_vals

    @parameter_values.setter
    def parameter_values(self, values):
        pass

class Sequential(Container):

    def __init__(self):
        super(Sequential, self).__init__()

    def _update_output(self, inp):

        last_output = self._prpare_inputs(inp)
        for m in self._modules:
            last_output = m(last_output)

        return last_output
