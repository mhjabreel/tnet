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
from theano.compile import shared_constructor
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import HostFromGpu, GpuFromHost
from tnet.core.cuda import _Variable as CudaVariable

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config



class _var(object):
    def __repr__(self):
        value = self.container.value
        t = str(value.dtype)
        t = t[0].upper() + t[1:]
        size = value.shape
        value = str(value)
        return value + "\n" + str(t) + "Tensor of size " + str(size)

    def __str__(self):

        return self.__repr__()

    @property
    def data(self):
        return self.get_value()


class _Variable(_var, theano.tensor.sharedvar.TensorSharedVariable):
    """docstring for Variable."""
    def __init__(self, value, name=None):

        if isinstance(value, list):
            pass
        elif isinstance(value, np.ndarray):
            broadcastable = (False,) * len(value.shape)
            t_type = TensorType(value.dtype, broadcastable=broadcastable)


        super(_Variable, self).__init__(type=t_type,
                                value=value,
                                name=name,
                                strict=False,
                                allow_downcast=True)


    def cuda(self):
        v = self.get_value()
        return CudaVariable(v, self.name)



@shared_constructor
def variable_shared_constructor(value, name=None):
    """
    tnet._Variable Constructor for TensorType.
    Notes
    -----
    Regarding the inference of the broadcastable pattern...
    The default is to assume that the value might be resized in any
    dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
    The optional `broadcastable` argument will override this default.
    """

    return _Variable(value=np.array(value), name=name)
