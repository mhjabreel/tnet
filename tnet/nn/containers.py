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
        return self

    @property
    def parameters(self):
        params = []
        for m in self._modules:
            p = m.parameters
            if not p is None:
                #params.append(p)
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


    def _set_running_mode(self, mode):

        self._running_mode = mode

        for m in self._modules:
            m._set_running_mode(mode)

class Sequential(Container):

    def __init__(self):
        super(Sequential, self).__init__()


    def _update_output(self, inp):

        self.input = []
        last_output = self._check_input(inp)
        self.input.append(last_output)
        for m in self._modules:
            last_output = m(last_output)
            self.input.append(last_output)

        return last_output


    @property
    def input_info(self):
        if len(self._modules) > 0:
            m = self._modules[0]
            return m.input_info

    def __rep__(self):

        tab = "  "
        next = " -> "
        line = '\n'
        s = "Sequential {\n" + tab + "[input"

        s1 = ""
        s2 = ""
        for i, m in enumerate(self._modules):
            s1 = s1 + next + "({})".format(i + 1)
            s2 = s2 + line + tab + "({}): {}".format(i + 1, m).replace(line, line + tab)

        s += s1 + next + "output]" +  s2 + line + "}"

        return s
