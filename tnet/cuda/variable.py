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

from theano.tensor.basic import _tensor_py_operators
from theano.compile import shared_constructor
from theano.sandbox.cuda.type import CudaNdarrayType
from tnet.core.var import _var
#from tnet.core.variable import SharedVariable as TensorVariable


T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config



class Variable(_var, theano.sandbox.cuda.var.CudaNdarraySharedVariable):

    def __init__(self, value, name=None, type=None):


        if isinstance(value, Variable):
            super(Variable, self).__init__(value.name, value.type, value=None, strict=None,
                             allow_downcast=None, container=value.container)
        else:

            if not isinstance(value, np.ndarray):
                raise TypeError()

            if type is None:
                broadcastable = (False,) * len(value.shape)
            type = CudaNdarrayType(broadcastable=broadcastable)
            super(Variable, self).__init__(name, type, value, strict=False,
                             allow_downcast=True, container=None)

    def clone(self):
        cp = self.__class__(self)
        cp.tag = copy.copy(self.tag)
        return cp

class Parameter(Variable):
    """docstring for Parameter."""
    def __init__(self, value, name=None, type=None):
        super(Parameter, self).__init__(value, name, type)
        gname = name + "_grad" if not name is None else None
        self.grad = Variable(np.array(self.container.value), gname)
        self.grad.zero()
