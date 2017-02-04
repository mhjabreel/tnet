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



class SharedVariable(_var, theano.sandbox.cuda.var.CudaNdarraySharedVariable):


    def float32(self):
        v = self.get_value()
        return TensorVariable(v, self.name)



@shared_constructor
def cuda_variable_shared_constructor(value, name=None, strict=False,
                            allow_downcast=None, borrow=False,
                            broadcastable=None, target='gpu'):
    """
    SharedVariable Constructor for CudaNdarrayType.
    """
    if target != 'gpu':
        raise TypeError('not for gpu')

    # THIS CONSTRUCTOR TRIES TO CAST VALUE TO A FLOAT32, WHICH THEN GOES ONTO THE CARD
    # SO INT shared vars, float64 shared vars, etc. all end up on the card.
    # THIS IS NOT THE DEFAULT BEHAVIOUR THAT WE WANT.
    # SEE float32_shared_constructor

    # TODO: what should strict mean in this context, since we always have to make a copy?
    if strict:
        _value = value
    else:
        _value = theano._asarray(value, dtype='float32')

    if not isinstance(_value, np.ndarray):
        raise TypeError('ndarray required')
    if _value.dtype.num != CudaNdarrayType.typenum:
        raise TypeError('float32 ndarray required')

    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
    type = CudaNdarrayType(broadcastable=broadcastable)
    #print("trying to return?")
    try:
        rval = SharedVariable(type=type, value=_value, name=name, strict=strict)
    except Exception as e:
        print("ERROR", e)
        raise
    return rval
