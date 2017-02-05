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
import theano.tensor as T
from theano.tensor.type import TensorType
from theano.gof import hashtype, Type, Variable
import numpy as np

__all__ = ["Delegator"]

class _delegator(TensorType):

    _delegators = 0

    """docstring for Delegator."""
    def __init__(self, dtype, shape=None, ndim=None, strict=False):
        # TODO: assert correct dtype
        # TODO: default dtype
        if shape is None and ndim is None:
            raise ValueError("Unknown delegator structur")
        if shape is not None:
            assert type(shape) in (list, tuple)
            ndim = len(shape)
        else:
            try:
                ndim = int(ndim)
            except Exception as e:
                print("Expected int value for ndim args found %s " % ndim)
                raise e
            strict = False # as we do not know the shape the stric arg is False by default

        self._shape = shape
        self._strict = strict


        super(_delegator, self).__init__(dtype, (False, ) * ndim)



    # override filter function to enable shape filtering
    def filter(self, data, strict=False, allow_downcast=None):

        strict = self._strict and strict
        if type(data) != np.ndarray:
            if strict:
                data = np.array(data)
            else:
                try:
                    data = np.array(data, self.dtype)
                except Exception as e:
                    print("%s Unable to cast to ndarray with type %s" % (self, self.dtype))
                    raise e

        if strict:
            if data.shape != self._shape:
                raise ValueError ("Cannot feed value of shape %s for Tensor %s, which has shape %s" % (data.shape, self, self._shape))

        return super(_delegator, self).filter(data, strict, allow_downcast)

    def __call__(self, name=None):
        if name is None:
            name = "Delegator:%d" % _delegator._delegators
            _delegator._delegators += 1

        return super(_delegator, self).__call__(name)

    def filter_variable(self, other, allow_convert=True):
        """
        Convert a symbolic Variable into a TensorType, if compatible.
        For the moment, only a TensorType, GpuArrayType and
        CudaNdarrayType will be
        converted, provided they have the same number of dimensions and
        dtype and have "compatible" broadcastable pattern.
        """
        if hasattr(other, '_as_TensorVariable'):
            other = other._as_TensorVariable()

        if not isinstance(other, Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = self.Constant(type=self, data=other)

        if other.type == self:
            return other

        if allow_convert:
            # Attempt safe broadcast conversion.
            other2 = self.convert_variable(other)
            print(other2)
            if other2 is not None and isinstance(other2.type, TensorType):
                return other2

        raise TypeError(
            'Delegatror: Cannot convert Type %(othertype)s '
            '(of Variable %(other)s) into Type %(self)s. '
            'You can try to manually convert %(other)s into a %(self)s.' %
            dict(othertype=other.type,
                 other=other,
                 self=self))

    def convert_variable(self, var):

        if ( isinstance(var.type, TensorType) and  # noqa
            self.dtype == var.type.dtype and
            self.ndim == var.type.ndim and
            all(sb == ob or ob for sb, ob in zip(self.broadcastable,
                                                 var.type.broadcastable))):
            v = T.patternbroadcast(var, self.broadcastable)
            print(v)
            return v

    def __str__(self):

        return "tnet.Delegator %s, dtype: %s, shape: %s, ndim: %s" %(self.name, self.dtype, self._shape, self.ndim)

    def clone(self, dtype=None, broadcastable=None):
        """
        Return a copy of the type optionally with a new dtype or
        broadcastable pattern.
        """

        if dtype is None:
            dtype = self.dtype
        if broadcastable is None:
            broadcastable = self.broadcastable
        ndim = len(broadcastable)
        return self.__class__(dtype, ndim=ndim)

def Delegator(dtype, shape=None, ndim=None, name=None, strict=False):
    if shape is None and ndim is None:
        raise ValueError("Unknown delegator structur")
    if shape is not None:
        assert type(shape) in (list, tuple)
        ndim = len(shape)
    else:
        try:
            ndim = int(ndim)
        except Exception as e:
            print("Expected int value for ndim args found %s " % ndim)
            raise e
    d = TensorType(dtype, (False, ) * ndim)(name)
    return d
