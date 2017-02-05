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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import math

from tnet.nn import Module, InputInfo

__all__ = [
    "Dropout",
    "SpatialDropout"
]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

class Dropout(Module):
    """docstring for Dropout."""
    def __init__(self, p, scale=True):
        self._p = p
        self._scale = scale
        super(Dropout, self).__init__()


    def _declare(self, **kwargs):
        pass

    def _update_output(self, inp):

        inp = super(Dropout, self)._update_output(inp)

        if not self.is_in_training or self._p == 0:
            return inp


        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        retain = 1. - self._p
        mask = rng.binomial(inp.shape, p=retain, dtype=inp.dtype)
        y = inp * mask
        if self._scale:
            y /= retain
        return y

    def __repr__(self):

        return "{}({})".format(self.__class__.__name__, self._p)

class SpatialDropout(Module):

    """
    This version performs the same function as nn.Dropout, however it assumes the 2 right-most dimensions of the input are spatial,
    performs one Bernoulli trial per output feature when training, and extends this dropout value across the entire feature map.

    As described in the paper "Efficient Object Localization Using Convolutional Networks" (http://arxiv.org/abs/1411.4280),
    if adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers)
        then iid dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease.
    In this case, nn.SpatialDropout will help promote independence between feature maps and should be used instead.

    nn.SpatialDropout accepts 3D or 4D inputs. If the input is 3D than a layout of (features x height x width) is assumed and for 4D (batch x features x height x width) is assumed.
    """
    def __init__(self, p, scale=True):
        self._p = p
        self._scale = scale
        super(SpatialDropout, self).__init__()


    def _declare(self, **kwargs):
        pass

    def _update_output(self, inp):

        inp = super(SpatialDropout, self)._update_output(inp)

        if not self.is_in_training or self._p == 0:
            return inp

        input_shape = inp.shape
        if inp.ndim == 3:
            noise_shape = (input_shape[0], 1, input_shape[2])
        elif inp.ndim == 4:
            noise_shape = (input_shape[0], input_shape[1], 1, 1)
        else:
            raise



        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        retain = 1. - self._p
        mask = rng.binomial(noise_shape, p=retain, dtype=inp.dtype)
        mask = T.patternbroadcast(mask, [dim == 1 for dim in noise_shape])
        y = inp * mask
        if self._scale:
            y /= retain
        return y

    def __repr__(self):

        return "{}({})".format(self.__class__.__name__, self._p)
