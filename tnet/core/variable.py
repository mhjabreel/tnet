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


import theano
import theano.tensor.basic
from theano.tensor.basic import TensorType, _tensor_py_operators
from theano.compile import SharedVariable
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.compile import shared_constructor
#from tnet.core.cuda import SharedVariable as CudaVariable

__all__ = []
try:
    # We must do those import to be able to create the full doc when nvcc
    # is not available

    from theano.sandbox.cuda.basic_ops import HostFromGpu, GpuFromHost
except ImportError:
    pass
T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

from tnet.core.var import _var

"""
Provide a simple user friendly API to Theano-managed memory.

"""

@staticmethod
def get_tensor_type(value):
    assert isinstance(value, np.ndarray), "Expected numpy.ndarray value got %s" % value
    broadcastable = (False,) * len(value.shape)
    type = TensorType(value.dtype, broadcastable=broadcastable)
    return type

class Variable(_var, SharedVariable):

    # default_update
    # If this member is present, its value will be used as the "update" for
    # this Variable, unless another update value has been passed to "function",
    # or the "no_default_updates" list passed to "function" contains it.

    def __init__(self, value, name=None, type=None):


        if isinstance(value, Variable):
            super(Variable, self).__init__(value.name, value.type, value=None, strict=None,
                             allow_downcast=None, container=value.container)
        else:

            if not isinstance(value, np.ndarray):
                raise TypeError()

            if type is None:
                type = Variable.get_type(value)
            super(Variable, self).__init__(name, type, value, strict=False,
                             allow_downcast=True, container=None)

Variable.get_type = get_tensor_type


class Parameter(Variable):
    """docstring for Parameter."""
    def __init__(self, value, name=None, type=None):
        super(Parameter, self).__init__(value, name, type)
        gname = name + "_grad" if not name is None else None
        self.grad = Variable(np.array(self.container.value), gname)
        self.grad.zero()
