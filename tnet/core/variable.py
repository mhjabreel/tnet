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
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import HostFromGpu, GpuFromHost

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

__all__ = []

class Variable(theano.tensor.sharedvar.TensorSharedVariable):
    """docstring for Variable."""
    def __init__(self, value, name=None):

        if isinstance(value, list):
            pass
        elif isinstance(value, np.ndarray):
            broadcastable = (False,) * len(value.shape)
            t_type = TensorType(value.dtype, broadcastable=broadcastable) \
                if theano.config.device == "cpu" else CudaNdarrayType(broadcastable=broadcastable)


        super(Variable, self).__init__(type=t_type,
                                value=value,
                                name=name,
                                strict=False,
                                allow_downcast=True)






    def __repr__(self):

        value = self.get_value()
        t = str(value.dtype)
        t = t[0].upper() + t[1:]
        size = value.shape

        value = str(value)


        return value + "\n" + str(t) + "Tensor of size " + str(size)




    def zero_like(self, name=None):


        return theano.shared(self.value * 0., name)

    def clone(self):
        cp = self.__class__(self.get_value())
        cp.tag = copy.copy(self.tag)
        return cp

    @property
    def value(self):
        return self.get_value()





class Parameter(Variable):
    """docstring for DifferentiableVariable."""
    def __init__(self, value, name=None):
        super(Parameter, self).__init__(value, name)
        gname = name + "_grad" if not name is None else None
        self.grad = self.zero_like(gname)#theano.shared(value * 0., gname)
