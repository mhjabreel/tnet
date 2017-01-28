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
    "TemporalConvolution"
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

        #stdv = 1 / math.sqrt(self._kw * self._kh * self._n_input_plane)
        stdv = np.sqrt(6. / (self._n_input_plane + self._n_output_plane))
        self._W_values = np.array(np.random.uniform(low=-stdv,
                                              high=stdv,
                                              size=self._filter_shape),
                                              theano.config.floatX)

        self._W = theano.shared(self._W_values, borrow=True)

        if self._has_bias:


            self._b_values = np.array(np.zeros(self._n_output_plane),
                                                      theano.config.floatX)

            self._b = theano.shared(self._b_values, borrow=True)


    def _update_output(self, inp):

        inp = self._prpare_inputs(inp)

        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)

        y = T.nnet.conv2d(inp, self._W,
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




class TemporalConvolution(SpatialConvolution):

    """
    Applies a 1D convolution over an input sequence composed of nInputFrame frames.
    The input tensor in forward(input) is expected to be a 3D tensor (batch_szie, sequence_length x input_frame_size).

    The parameters are the following:

        input_frame_size: The input frame size expected in sequences given into forward().
        output_frame_size: The output frame size the convolution layer will produce.
        kw: The kernel width of the convolution
        dw: The step of the convolution. Default is 1.
        bias: whether to include a bias. Default is True.

        Here is a simple example:

        inp = 5  #dimensionality of one sequence element
        outp = 1 #number of derived features for one sequence element
        kw = 1   #kernel only operates on one sequence element per step



        >> from tnet import nn
        >> import numpy as np

        >> tconv = nn.TemporalConvolution(inp, outp, kw)
        >> x = np.random.rand(2, 7, inp) # two sequences of 7 elements
        >> out = tconv.forward(x)
        >> print(out.shape)

        which gives:
        (2, 1, 7)
    """

    def __init__(self, input_frame_size, output_frame_size, kw, dw=1, bias=True):

        self._input_info = InputInfo(dtype=config.floatX, shape=[kw + 1, input_frame_size])
        self._input_frame_size = input_frame_size
        self._output_frame_size = output_frame_size
        self._kw = kw
        self._dw = dw
        self._has_bias = bias


        super(TemporalConvolution, self).__init__(1, output_frame_size, kw, input_frame_size, dw, bias=bias)
        self._image_shape = (None, self._n_input_plane, None, input_frame_size)


    def _update_output(self, inp):

        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)
        if inp.ndim == 3:
            inp = inp.dimshuffle(0, 'x', 1, 2)

        out = super(TemporalConvolution, self)._update_output(inp)
        out = T.reshape(out, (out.shape[0], out.shape[2], out.shape[3]))
        return out
