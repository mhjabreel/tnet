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
from tnet.core.cuda import SharedVariable as CudaVariable

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

from tnet.core.var import _var


class SharedVariable(_var, theano.tensor.sharedvar.TensorSharedVariable):


    def cuda(self):
        v = self.get_value()
        return CudaVariable(v, self.name)


@shared_constructor
def variable_shared_constructor(value, name=None, strict=False, allow_downcast=None,
                       borrow=False, broadcastable=None, target='cpu'):
    """
    SharedVariable Constructor for TensorType.
    Notes
    -----
    Regarding the inference of the broadcastable pattern...
    The default is to assume that the value might be resized in any
    dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
    The optional `broadcastable` argument will override this default.
    """
    if target != 'cpu':
        raise TypeError('not for cpu')

    if not isinstance(value, np.ndarray):
        raise TypeError()

    # if no broadcastable is given, then the default is to assume that
    # the value might be resized in any dimension in the future.
    #
    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
    type = TensorType(value.dtype, broadcastable=broadcastable)
    return SharedVariable(type=type,
                                value=np.array(value, copy=(not borrow)),
                                name=name,
                                strict=strict,
                                allow_downcast=allow_downcast)
