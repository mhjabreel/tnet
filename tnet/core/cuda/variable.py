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

from tnet.core.variable import _var, _Variable as TensorVariable

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config



class _Variable(_var, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
    """docstring for Variable."""
    def __init__(self, value, name=None):

        if isinstance(value, list):
            pass
        elif isinstance(value, np.ndarray):
            broadcastable = (False,) * len(value.shape)
            t_type = CudaNdarrayType(broadcastable)


        super(_Variable, self).__init__(type=t_type,
                                value=value,
                                name=name,
                                strict=False,
                                allow_downcast=True)


    def float32(self):
        v = self.get_value()
        return TensorVariable(v, self.name)



@shared_constructor
def cuda_variable_shared_constructor(value, name=None):
    """
    tnet.cuda._Variable Constructor for TensorType.
    Notes
    -----
    Regarding the inference of the broadcastable pattern...
    The default is to assume that the value might be resized in any
    dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
    The optional `broadcastable` argument will override this default.
    """

    return _Variable(value=np.array(value), name=name)
