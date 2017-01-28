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
from __future__ import division
from __future__ import print_function

import numpy as np
import theano
import math

from tnet.nn import Module, InputInfo

__all__ = [
    "Flatten",
    "Max",
    "Min",
    "Mean",
    "Sum"
]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config


class _GobalPooling(Module):
    def __init__(self, pool_fn, dimension, n_input_dim=None):


        self._dimension = dimension
        self._n_input_dim = n_input_dim
        self._pool_fn = pool_fn
        super(Max, self).__init__()


    def _get_positive_index(self, inp):
        d = self._dimension
        if  d < 0:
            d = inp.ndim + d + 1
        elif not self.n_input_dim is None and inp.ndim == (self.n_input_dim + 1):
            d = d + 1
        return d

    def _declare(self):
        pass


    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)
        d = self._get_positive_index(inp)
        return self._pool_fn(inp, axis=d)


class Max(_GobalPooling):

    """
    Applies a max operation over dimension dimension.
    The parameters are the following:

    Hence, if an nxpxq Tensor was given as input, and dimension = 2 then an nxq matrix would be output.
    When n_input_dim is provided, inputs larger than that value will be considered batches
        where the actual dimension to apply the max operation will be dimension dimension + 1.
    """

    def __init__(self, dimension, n_input_dim=None):

        super(Max, self).__init__(T.max, dimension, n_input_dim)


class Min(_GobalPooling):

    """
    Applies a min operation over dimension dimension.
    The parameters are the following:

    Hence, if an nxpxq Tensor was given as input, and dimension = 2 then an nxq matrix would be output.
    When n_input_dim is provided, inputs larger than that value will be considered batches
        where the actual dimension to apply the max operation will be dimension dimension + 1.
    """

    def __init__(self, dimension, n_input_dim=None):

        super(Max, self).__init__(T.min, dimension, n_input_dim)

class Mean(_GobalPooling):

    """
    Applies a mean operation over dimension dimension.
    The parameters are the following:

    Hence, if an nxpxq Tensor was given as input, and dimension = 2 then an nxq matrix would be output.
    When n_input_dim is provided, inputs larger than that value will be considered batches
        where the actual dimension to apply the max operation will be dimension dimension + 1.
    """

    def __init__(self, dimension, n_input_dim=None):

        super(Max, self).__init__(T.mean, dimension, n_input_dim)


class Sum(_GobalPooling):

    """
    Applies a max operation over dimension dimension.
    The parameters are the following:

    Hence, if an nxpxq Tensor was given as input, and dimension = 2 then an nxq matrix would be output.
    When n_input_dim is provided, inputs larger than that value will be considered batches
        where the actual dimension to apply the max operation will be dimension dimension + 1.
    """

    def __init__(self, dimension, n_input_dim=None):

        super(Max, self).__init__(T.sum, dimension, n_input_dim)

class Flatten(Module):

    def __init__(self):

        super(Flatten, self).__init__()


    def _declare(self):
        pass


    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)

        y = T.reshape(inp, (inp.shape[0], T.prod(inp.shape) // inp.shape[0]))

        return y
