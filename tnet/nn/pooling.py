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
    "SpatialMaxPooling",

]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config


class _SpatialPooling(Module):

    def __init__(self, pool_mode, kw, kh, dw=None, dh=None, padw=0, padh=0):

        self._kh = kh
        self._kw = kw

        if dw is None:
            dw = kw

        self._dw = dw

        if dh is None:
            dh = kh

        self._dh = dh

        self._padw = padw
        self._padh = padh

        self._ciel_mode = False

        self._pool_mode = pool_mode

        self._input_info = InputInfo(dtype=config.floatX, shape=[1, kh + 1, kw + 1])

        super(_SpatialPooling, self).__init__()

    def _update_output(self, inp):

        inp = super(_SpatialPooling, self)._update_output(inp)

        ds = (self._kh, self._kw)
        st = (self._dh, self._dw)
        ignore_border = not self._ciel_mode
        padding = (self._padh, self._padw)
        mode = self._pool_mode


        y = T.signal.pool.pool_2d(inp,
                          ds,
                          ignore_border,
                          st,
                          padding,
                          mode)
        return y

    def ciel(self):
        self._ciel_mode = True
        return self

    def floor(self):
        self._ciel_mode = False
        return self


class SpatialMaxPooling(_SpatialPooling):

    def __init__(self, kw, kh, dw=None, dh=None, padW=0, padH=0):
        super(SpatialMaxPooling, self).__init__('max', kw, kh, dw, dh, padW, padH)
