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

from tnet.nn import Module, InputInfo

__all__ = [
    "Container",
    "Bottle",
    "Sequential",
    "Parallel",
    "Concat"
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
        self._params += module.parameters
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

class Bottle(Container):
    """
    Bottle allows varying dimensionality input to be forwarded through
        any module that accepts input of nb_input_dim dimensions, and generates output of nOutputDim dimensions.

    Bottle can be used to forward a 4D input of varying sizes through a 2D module b x n.
    The module Bottle(module, 2) will accept input of shape p x q x r x n and outputs with the shape p x q x r x m.
    Internally Bottle will view the input of module as p*q*r x n, and view the output as p x q x r x m.
    The numbers p x q x r are inferred from the input and can change for every forward/backward pass.
    """
    def __init__(self, module, nb_input_dim=2, nb_out_dim=None):

        super(Bottle, self).__init__()
        self._module = module
        self._nb_input_dim = int(nb_input_dim)
        self._nb_out_dim = self._nb_input_dim if nb_out_dim is None else int(nb_out_dim)

    def _update_output(self, inp):

        inp = self._check_input(inp)
        pass



    @property
    def input_info(self):
        if len(self._modules) > 0:
            m = self._modules[0]
            return m.input_info

    def __repr__(self):

        tab = "  "
        next = " -> "
        line = '\n'
        s = "Bottle {\n" + tab + "[input" + next + "(1) output]" + line

        s += tab + "(1): " + str(self._module).replace(line, line + tab)
        s +=  line + "}"
        return s


class Sequential(Container):

    def __init__(self):
        super(Sequential, self).__init__()


    def _update_output(self, inp):


        last_output = self._check_input(inp)

        for m in self._modules:
            last_output = m(last_output)


        return last_output


    @property
    def input_info(self):
        if len(self._modules) > 0:
            m = self._modules[0]
            return m.input_info

    def __repr__(self):

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


class Parallel(Container):
    """
    A container module that applies its ith child module to the ith slice of the input by using select on dimension dim.
    It concatenates the results of its contained modules together along dimension out_dim.

    Example:

    >>> m = nn.Parallel(1,0);   # Parallel container will associate a module to each slice of dimension 1
                                # (column space), and concatenate the outputs over the 1st dimension.

    >>> m.add(nn.Linear(10,3))  # Linear module (input 10, output 3), applied on 1st slice of dimension 1
    >>> m.add(nn.Linear(10,2))    # Linear module (input 10, output 2), applied on 2nd slice of dimension 1

    >>> m.forward(tnet.randn(10, 2))
    Which gives the output:
    [ 2.39241979  0.29238148 -1.52912492  1.21551019 -0.8211203 ]
    float32_tensor of size (5L,)
    """
    def __init__(self, dim, out_dim):
        self._dim = dim
        self._out_dim = out_dim
        super(Parallel, self).__init__()

    def _update_output(self, inp):


        inp = self._check_input(inp)
        inp_ = inp.swapaxes(self._dim, 0)
        outputs = []
        for i, m in enumerate(self._modules):
            out = m(inp_[i])
            outputs.append(out)

        return T.concatenate(outputs, axis=self._out_dim)


    @property
    def input_info(self):
        if len(self._modules) > 0:
            shape = [None] * self._dim + 1
            type = self._modules[0].input_info.dtype
            return InputInfo(shape, type)

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = '  |`-> '
        ext = '  |    '
        extlast = '       '
        last = '   ... -> '
        res = "Parallel"
        res += ' {' + line + tab + 'input'
        for i in range(len(self._modules)):
            if i == len(self._modules) - 1:
                res += line + tab + next + '(' + str(i + 1) + '): ' + \
                    str(self._modules[i]).replace(line, line + tab + extlast)
            else:
                res += line + tab + next + '(' + str(i + 1) + '): ' + str(self._modules[i]).replace(line, line + tab + ext)

        res += line + tab + last + 'output'
        res += line + '}'
        return res

class Concat(Container):
    """
    A cotiner that concatenates the output of one layer of "parallel" modules along the provided dimension dim:
        hey take the same inputs, and their output is concatenated.
    Example:

    >>> m = nn.Concat(0)
    >>> m.add(nn.Linear(5, 3))
    >>> m.add(nn.Linear(5, 7))

    >>> m.forward(tnet.randn(10, 2))
    Which gives the output:
    [-1.04109108  0.98187275  1.16660643 -0.71731795 -0.85827608  0.44558068 -0.70544224]
    float32_tensor of size (7L,)
    """
    def __init__(self, dim):
        self._dim = dim

        super(Concat, self).__init__()

    def _update_output(self, inp):


        inp = self._check_input(inp)

        outputs = []
        for i, m in enumerate(self._modules):
            out = m(inp)
            outputs.append(out)

        return T.concatenate(outputs, axis=self._dim)


    @property
    def input_info(self):
        if len(self._modules) > 0:
            shape = [None] * self._dim + 1
            type = self._modules[0].input_info.dtype
            return InputInfo(shape, type)

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = '  |`-> '
        ext = '  |    '
        extlast = '       '
        last = '   ... -> '
        res = "Concat"
        res += ' {' + line + tab + 'input'
        for i in range(len(self._modules)):
            if i == len(self._modules) - 1:
                res += line + tab + next + '(' + str(i + 1) + '): ' + \
                    str(self._modules[i]).replace(line, line + tab + extlast)
            else:
                res += line + tab + next + '(' + str(i + 1) + '): ' + str(self._modules[i]).replace(line, line + tab + ext)

        res += line + tab + last + 'output'
        res += line + '}'
        return res


# TODO
class DepthConcat(Container):

    pass
