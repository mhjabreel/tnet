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
