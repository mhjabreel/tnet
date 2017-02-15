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

import tnet
import numpy as np
import from tnet.nn
from tnet.nn import Module, InputInfo


class RNNState(object):

    def __init__(self, rnn_size, nb_states=1, dtype=None, default_batch_size=None):

        self._rnn_size = rnn_size
        self._buffers = None
        self._nb_states = nb_states
        
        if dtype is None:
            dtype = tnet.default_dtype()

        self._dtype = dtype



    def reset(self, batch_size=1):
        self._buffers = []
        for i in range(self._nb_states):
            v = np.zeros((batch_size, self._rnn_size), dtype=self._dtype)
            v[...] = self._init_val

            self._buffers.append(tnet.Variable(v))



    def __getitem__(self, key):
        if self._buffers is None:
            raise ValueError("Please call function reset first")

        key = int(key)
        return self._buffers[key]

    def __iter__(self):
        k = 0
        while k < len(self._buffers):
            yield self._buffers[k]
            k += 1






class RNNCell(Module):
    """docstring for RNNCell."""
    def __init__(self, input_dim, rnn_size):

        self._rnn_size = rnn_size
        self._input_dim = input_dim

        super(RNNCell, self).__init__()

    def _declare(self, **kwargs):
        pass

    def _compile(self, **kwargs):
        pass


class BasicCell(RNNCell):

    def __init__(self, input_dim, rnn_size):
        self._state = RNNState(rnn_size)
        super(BasicCell, self).__init__(input_dim, rnn_size)

    def _declare(self, **kwargs):

        self._i2h_linear = nn.Linear(self._input_dim, self._rnn_size, False)
        self._h2h_linear = nn.Linear(self._rnn_size, self._rnn_size)

        self._params = self._i2h_linear.parameters + self._h2h_linear.parameters


    def _update_output(self, inp):

        inp = self._check_input(inp)
        bsz = inp.shape[0]








class LSTMCell(RNNCell):
    """docstring for LSTMCell."""
    def __init__(self, arg):
        super(LSTMCell, self).__init__()
        self.arg = arg


class GRUCell(object):
    """docstring for GRUCell."""
    def __init__(self, arg):
        super(GRUCell, self).__init__()
        self.arg = arg
