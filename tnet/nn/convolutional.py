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
    "SpatialConvolution",
]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config


class SpatialConvolution(Module):

    def __init__(self, n_input_plane, n_output_plane, kw, kh, dw=1, dh=1, padw=0, padh=0, bias=True):

        self._input_info = InputInfo(dtype=config.floatX, shape=[n_input_plane, kh + 1, kw + 1])
        self._n_input_plane = n_input_plane
        self._n_output_plane = n_output_plane
        self._kw = kw
        self._kh = kh
        self._dw = dw
        self._dh = dh
        self._padw = padw
        self._padh = padh
        self._has_bias = bias

        super(SpatialConvolution, self).__init__()


    def _declare(self):

        self._filter_shape = (self._n_output_plane, self._n_input_plane, self._kh, self._kw)
        self._image_shape = (None, self._n_input_plane, None, None)

        stdv = 1 / math.sqrt(self._kw * self._kh * self._n_input_plane)

        self._W_values = np.array(np.random.uniform(low=-stdv,
                                              high=stdv,
                                              size=self._filter_shape),
                                              theano.config.floatX)

        self._W = theano.shared(self._W_values, borrow=True)

        if self._has_bias:

            self._b_values = np.array(np.random.uniform(low=-stdv,
                                                      high=stdv,
                                                      size=(self._n_output_plane)),
                                                      theano.config.floatX)

            self._b = theano.shared(self._b_values, borrow=True)


    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)

        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)

        y = T.nnet.conv2d(inp,
                          self._W,

                          subsample=(self._dh, self._dw),
                          border_mode=(self._padh, self._padw))

        if self._has_bias:
            y += self._b.dimshuffle('x', 0, 'x', 'x')

        return y


    @property
    def parameters(self):
        if self._has_bias:
            return [self._W, self._b]
        return [self._W]

    @property
    def parameter_values(self):
        if self._has_bias:
            return [self._W_values, self._b_values]
        return [self._W_values]

    @parameter_values.setter
    def parameter_values(self, values):
        self._W_values = values[0]
        if self._has_bias:
            self._b_values = values[1]
