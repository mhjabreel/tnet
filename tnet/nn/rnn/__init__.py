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
from tnet.nn import (Module, InputInfo,
                     Container, Sigmoid, Tanh, ReLU,
                     CAddList, CMulList, CSubList, Linear,
                     Narrow, Transpose, Squeeze, Unsqueeze, MM)


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

    def copy(self, vals):
        pass


class RNNCell(Module):
    """docstring for RNNCell."""

    def __init__(self, input_dim, rnn_size):
        self._rnn_size = rnn_size
        self._input_dim = input_dim
        self._buffer_state = None
        self._input_info = InputInfo(shape=[input_dim], dtype=tnet.default_dtype())
        super(RNNCell, self).__init__()

    def _declare(self, **kwargs):
        pass

    def save_last_state(self, state):
        pass

    def zero_state(self, batch_size):
        pass

    @property
    def rnn_size(self):
        return self._rnn_size

    @property
    def input_dim(self):
        return self._input_dim

    def save_state(self, state):
        self._buffer_state = state

    @property
    def state(self):
        return self._buffer_state


class BasicCell(RNNCell):
    def _declare(self, **kwargs):
        self._i2h_linear = Linear(self._input_dim, self._rnn_size)
        self._h2h_linear = Linear(self._rnn_size, self._rnn_size, False)

        self._params = self._i2h_linear.parameters + self._h2h_linear.parameters

    def _update_output(self, inp):
        inp = self._check_input(inp)
        x = inp[0]
        h = inp[1]
        i2h = self._i2h_linear(x)
        h2h = self._h2h_linear(h)
        h = CAddList()([i2h, h2h])
        return h

    def zero_state(self, batch_size):
        state = tnet.T.zeros((batch_size, self._rnn_size))
        state = tnet.T.unbroadcast(state, 1)
        self._buffer_state = [state]


class ElmanCell(RNNCell):
    def _declare(self, **kwargs):
        self._i2h_linear = Linear(self._input_dim, self._rnn_size)
        self._h2h_linear = Linear(self._rnn_size, self._rnn_size, False)

        self._params = self._i2h_linear.parameters + self._h2h_linear.parameters

    def _update_output(self, inp):
        inp = self._check_input(inp)
        x = inp[0]
        h = inp[1]
        i2h = Sigmoid()(self._i2h_linear(x))
        h2h = self._h2h_linear(h)
        h = CAddList()([i2h, h2h])
        return h

    def zero_state(self, batch_size):
        self._buffer_state = [tnet.T.zeros((batch_size, self._rnn_size))]


class TanhCell(RNNCell):
    def _declare(self, **kwargs):
        self._i2h_linear = Linear(self._input_dim, self._rnn_size, False)
        self._h2h_linear = Linear(self._rnn_size, self._rnn_size)

        self._params = self._i2h_linear.parameters + self._h2h_linear.parameters

    def _update_output(self, inp):
        inp = self._check_input(inp)
        x = inp[0]
        h = inp[1]
        i2h = Tanh()(self._i2h_linear(x))
        h2h = self._h2h_linear(h)
        h = CAddList()([i2h, h2h])
        return h

    def zero_state(self, batch_size):
        self._buffer_state = [tnet.T.zeros((batch_size, self._rnn_size))]


class ReLUCell(RNNCell):
    def _declare(self, **kwargs):
        self._i2h_linear = Linear(self._input_dim, self._rnn_size, False)
        self._h2h_linear = Linear(self._rnn_size, self._rnn_size)

        self._params = self._i2h_linear.parameters + self._h2h_linear.parameters

    def _update_output(self, inp):
        inp = self._check_input(inp)
        x = inp[0]
        h = inp[1]
        i2h = ReLU()(self._i2h_linear(x))
        h2h = self._h2h_linear(h)
        h = CAddList()([i2h, h2h])
        return h

    def zero_state(self, batch_size):
        self._buffer_state = [tnet.T.zeros((batch_size, self._rnn_size))]


class LSTMCell(RNNCell):
    def _declare(self, **kwargs):
        self._i2h_linear = Linear(self._input_dim, 4 * self._rnn_size, False)
        self._h2h_linear = Linear(self._rnn_size, 4 * self._rnn_size)

        self._params = self._i2h_linear.parameters + self._h2h_linear.parameters

    def _update_output(self, inp):

        inp = self._check_input(inp)

        x = inp[0]
        prev_h = inp[1]
        prev_c = inp[2]

        x2h = self._i2h_linear(x)
        h2h = self._h2h_linear(prev_h)

        preactivations = CAddList()([x2h, h2h])
        # gates
        pre_sigmoid_chunk = Narrow(1, 0, 3 * self._rnn_size)(preactivations)
        all_gates = Sigmoid()(pre_sigmoid_chunk)

        in_gate = Narrow(1, 0, self._rnn_size)(all_gates)
        forget_gate = Narrow(1, self._rnn_size, self._rnn_size)(all_gates)
        out_gate = Narrow(1, 2 * self._rnn_size, self._rnn_size)(all_gates)

        # input
        in_chunk = Narrow(1, 3 * self._rnn_size, self._rnn_size)(preactivations)
        in_transform = Tanh()(in_chunk)

        # previous cell state contribution
        c_forget = CMulList()([forget_gate, prev_c])

        # input contribution
        c_input = CMulList()([in_gate, in_transform])

        next_c = CAddList()([c_forget, c_input])

        c_transform = Tanh()(next_c)
        next_h = CMulList()([out_gate, c_transform])

        return next_h, next_c

    def zero_state(self, batch_size):
        self._buffer_state = [tnet.T.zeros((batch_size, self._rnn_size)), tnet.T.zeros((batch_size, self._rnn_size))]


class GRUCell(RNNCell):
    def _declare(self, **kwargs):

        self._i2h_linear = Linear(self._input_dim, 3 * self._rnn_size, False)
        self._h2h_linear = Linear(self._rnn_size, 3 * self._rnn_size, False)

        self._params = self._i2h_linear.parameters + self._h2h_linear.parameters

    def _update_output(self, inp):

        nhid = self._rnn_size

        inp = self._check_input(inp)
        x = inp[0]
        prev_h = inp[1]

        x2h = self._i2h_linear(x)  # bsz x (3 * rnn_size)
        h2h = self._h2h_linear(prev_h)

        ru = CAddList()([
            Narrow(1, 0, 2 * nhid)(h2h ),
            Narrow(1, 0, 2 * nhid)(x2h)
        ])  # bsz x (2 * rnn_size)

        # gates
        ugate = Sigmoid()(Narrow(1, 0, nhid)(ru))
        rgate = Sigmoid()(Narrow(1, nhid, nhid)(ru))

        output = Tanh()(
            CAddList()([
                Narrow(1, 2 * nhid, nhid)(x2h),
                CMulList()([
                    rgate,
                    Narrow(1, 2 * nhid, nhid)(h2h)
                ])
            ])
        )

        next_h = CAddList()([
            output,
            CMulList()([
                ugate,
                CSubList()([
                    prev_h,
                    output
                ])
            ])
        ])

        return next_h

    def zero_state(self, batch_size):
        state = tnet.T.zeros((batch_size, self._rnn_size))
        state = tnet.T.unbroadcast(state, 1)
        self._buffer_state = [state]


class Recurrent(Container):
    def __init__(self,
                 rnn_cell,
                 nb_layers=1,
                 return_sequences=True):
        super(Recurrent, self).__init__()
        self._rnn_cell = rnn_cell
        self._nb_layers = nb_layers
        self._return_sequences = return_sequences

        self.add(rnn_cell)

        self._extend()

    def _extend(self):

        for _ in range(1, self._nb_layers):
            c = self._rnn_cell.__class__(self._rnn_cell.rnn_size, self._rnn_cell.rnn_size)
            self.add(c)

    def _update_output(self, inp):

        def sequence(m, seq_inp):

            ndim = seq_inp.ndim

            if ndim == 2:
                bsz = 1
                seq_inp = Unsqueeze(1)(seq_inp)
            elif ndim == 3:
                bsz = seq_inp.shape[0]
                seq_inp = Transpose(1, 0, 2)(seq_inp)
            else:
                raise ValueError("Unexpected number of input dimesnsion expected 2D or 3D, got %dD" % ndim)

            m.zero_state(bsz)
            
            def step(*input_and_state):
                state_t = m(input_and_state)
                return state_t

            states, _ = tnet.theano.scan(step, sequences=seq_inp, outputs_info=m.state)
            if isinstance(states, list):
                states = [Transpose(1, 0, 2)(s) for s in states]
            else:
                states = Transpose(1, 0, 2)(states)

            # m.save_state(states[:, -1, :]) # for statefull

            if ndim == 2:
                if isinstance(states, list):
                    states = [Squeeze(0)(s) for s in states]
                else:
                    states = Squeeze(0)(states)
            return states

        last_output = self._check_input(inp)
        outs = []
        for m in self._modules:

            outs_m = sequence(m, last_output)
            if isinstance(outs_m, list):
                last_output = outs_m[0]
            else:
                last_output = outs_m
            outs += [last_output]

        return outs

    @property
    def input_info(self):
        if len(self._modules) > 0:
            m = self._modules[0]
            return m.input_info


class SimpleRNN(Recurrent):
    def __init__(self, input_dim, rnn_size, nb_layers=1, return_sequences=True):
        cell = BasicCell(input_dim, rnn_size)
        super(SimpleRNN, self).__init__(cell, nb_layers, return_sequences)


class ElmanRNN(Recurrent):
    def __init__(self, input_dim, rnn_size, nb_layers=1, return_sequences=True):
        cell = ElmanCell(input_dim, rnn_size)
        super(ElmanRNN, self).__init__(cell, nb_layers, return_sequences)


class TanhRNN(Recurrent):
    def __init__(self, input_dim, rnn_size, nb_layers=1, return_sequences=True):
        cell = TanhCell(input_dim, rnn_size)
        super(TanhRNN, self).__init__(cell, nb_layers, return_sequences)


class ReLURNN(Recurrent):
    def __init__(self, input_dim, rnn_size, nb_layers=1, return_sequences=True):
        cell = ReLUCell(input_dim, rnn_size)
        super(ReLURNN, self).__init__(cell, nb_layers, return_sequences)


class LSTM(Recurrent):
    def __init__(self, input_dim, rnn_size, nb_layers=1, return_sequences=True):
        cell = LSTMCell(input_dim, rnn_size)
        super(LSTM, self).__init__(cell, nb_layers, return_sequences)


class GRU(Recurrent):
    def __init__(self, input_dim, rnn_size, nb_layers=1, return_sequences=True):
        cell = GRUCell(input_dim, rnn_size)
        super(GRU, self).__init__(cell, nb_layers, return_sequences)


