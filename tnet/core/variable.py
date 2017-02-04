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

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

from tnet.core.var import _var

"""
Provide a simple user friendly API to Theano-managed memory.

"""

class Variable(_var, _tensor_py_operators, SharedVariable):

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
                broadcastable = (False,) * len(value.shape)
                type = TensorType(value.dtype, broadcastable=broadcastable)
            super(Variable, self).__init__(name, type, value, strict=False,
                             allow_downcast=True, container=None)




    def zero(self, borrow=True):
        """
        Set the values of a shared variable to 0.

        Parameters
        ----------
        borrow : bbol
            True to modify the value of a shared variable directly by using
            its previous value. Potentially this can cause problems
            regarding to the aliased memory.

        Changes done with this function will be visible to all functions using
        this SharedVariable.

        """
        if borrow:
            self.container.value[...] = 0.
        else:
            self.container.value = 0. * self.container.value


    def clone(self):
        cp = self.__class__(self)
        cp.tag = copy.copy(self.tag)
        return cp


    def cuda(self):

        if isinstance(self.type, CudaNdarrayType):
            return self

        _value = theano._asarray(self.data, dtype='float32')
        broadcastable = self.broadcastable
        if broadcastable is None:
            broadcastable = (False,) * len(_value.shape)
        type = CudaNdarrayType(broadcastable=broadcastable)

        return Variable(_value, self.name, type)

    def float32(self):

        if isinstance(self.type, TensorType) and self.dtype == np.float32:
            return self

        _value = theano._asarray(self.data, dtype='float32')
        broadcastable = self.broadcastable
        if broadcastable is None:
            broadcastable = (False,) * len(_value.shape)
        type = TensorType(_value.dtype, broadcastable=broadcastable)

        return Variable(_value, self.name, type)





class Parameter(Variable):
    """docstring for Parameter."""
    def __init__(self, value, name=None, type=None):
        super(Parameter, self).__init__(value, name, type)
        gname = name + "_grad" if not name is None else None
        self.grad = Variable(self.data, gname)
        self.grad.zero()

    def clone(self):
        cp = self.__class__(self, self.name)
        cp.tag = copy.copy(self.tag)
        return cp

#@shared_constructor
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

    var = None
    try:
        var =  Variable(value=np.array(value),
                                    name=name)
    except Exception as e:
        print e


    return var
