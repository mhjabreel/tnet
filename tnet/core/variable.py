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
import copy
import numpy as np
import math

import theano
import theano.tensor.basic
from theano.tensor.basic import TensorType, _tensor_py_operators
from theano.compile import shared_constructor, SharedVariable

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
shared = theano.shared
config = theano.config

__all__ = ["Variable", "DifferentiableVariable"]

class Variable(object):
    """docstring for Variable."""
    def __init__(self, value, name=None):
        """
        if isinstance(value, list):
            pass
        elif isinstance(value, np.ndarray):
            broadcastable = (False,) * len(value.shape)
            type = TensorType(value.dtype, broadcastable=broadcastable)

        super(Variable, self).__init__(type=type,
                                value=value,
                                name=name,
                                strict=True,
                                allow_downcast=True)
        """

        self.data = shared(value, name = name)
        self._value = value

    def __str__(self):

        value = self._value
        t = self._value .dtype
        size = self._value .shape

        value = str(value)


        return value + "\n" + str(t) + "_tensor of size " + str(size)




    def zero_like(self, name=None):

        return self.data.zero_like(name)



    def __call__(self):
        return self.data
        
    @property
    def value(self):
        return self._value

    @property
    def dtype(self):
        return self._value.dtype

    def __getattr__(self, name):
        #if name in self._fallthrough_methods:
        return getattr(self.data, name)
        #raise AttributeError(name)

    def __add__(self, other):
        if isinstance(other, Variable):
            return self.data + other.data
        else:
            return self.data + other

    def __sub__(self, other):
        if isinstance(other, Variable):
            return self.data - other.data
        else:
            return self.data - other




class DifferentiableVariable(Variable):
    """docstring for DifferentiableVariable."""
    def __init__(self, value, name=None):
        super(DifferentiableVariable, self).__init__(value, name)
        gname = name + "_grad" if not name is None else None

        self.grad = shared(np.zeros_like(self.value).astype(self.dtype)) #self.zero_like(gname)
