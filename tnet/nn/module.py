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

from abc import ABCMeta, abstractmethod

import tnet
import numpy as np
import theano
import math

__all__ = [
    "Module",
    "InputInfo"
]

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config




class RunningMode(object):
    """docstring for RunningMode."""
    TRAIN, EVAL = True, False


class InputInfo(object):

    def __init__(self, shape=None, dtype=None):

        self.dtype = dtype
        self.shape = shape

class Module(object):

    def __init__(self, **kwargs):

        if not hasattr(self, '_input_info'):
            self._input_info = None

        self._running_mode = RunningMode.EVAL

        self._declare(**kwargs)
        self._compile(**kwargs)


    def _compile(self, **kwargs):
        if hasattr(self, '_input_info') and not self._input_info is None:
            input_shape = self._input_info.shape

            input_shape = [s if not s is None else 1 for s in input_shape]

            mock_input = np.array(np.zeros([1] + input_shape), self._input_info.dtype)

            self.forward(mock_input)
        else:
            self.forward(np.random.rand(1) + 1.5)



    def _check_input(self, input_or_inputs):

        type_of_input = type(input_or_inputs)

        if type_of_input == list or type_of_input == tuple:

            type_of_inputs = [type(inp) for inp in input_or_inputs]


            all_types_are_coorect = all([ t == np.ndarray or \
                                         t == T.TensorVariable or \
                                         t == tnet.Variable or \
                                         t == T.TensorConstant \
                                            for t in type_of_inputs])

            if not all_types_are_coorect:
                raise  ValueError("Wrong types are passed")

            input_or_inputs = [tnet.Variable(inp) if isinstance(t, np.ndarray) else \
                               inp for inp in input_or_inputs]
        elif type_of_input == np.ndarray:
            input_or_inputs = to_tensor(input_or_inputs)
        else:
            if not (type_of_input == T.TensorVariable or type_of_input == T.TensorConstant or type_of_input == tnet.Variable):
                raise  ValueError("Wrong types are passed")

        return input_or_inputs

    def _update_output(self, input_or_inputs):

        input_or_inputs = self._check_input(input_or_inputs)
        assert isinstance(input_or_inputs, tnet.Variable) or \
               isinstance(input_or_inputs, T.TensorConstant) or \
               isinstance(input_or_inputs, T.TensorVariable)

        self.input = input_or_inputs

        return input_or_inputs




    def __call__(self, input_or_inputs):

        return self._update_output(input_or_inputs)


    def forward(self, input_or_inputs):
        out = self._update_output(input_or_inputs)
        if type(out) == list:
            print(out)
            self._output = [s.eval() for s in out]
        else:
            self._output = out.eval()

        return self._output





    def _set_running_mode(self, mode):
        self._running_mode = mode


    def training(self):
        self._set_running_mode(RunningMode.TRAIN)

    def evaluate(self):
        self._set_running_mode(RunningMode.EVAL)

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        return []

    @property
    def parameter_values(self):
        return []

    @parameter_values.setter
    def parameter_values(self, values):
        pass

    @property
    def running_mode(self):
        return self._runing_mode

    @property
    def is_in_training(self):
        return self._running_mode == RunningMode.TRAIN


    @property
    def input_info(self):
        if hasattr(self, '_input_info'):
            return self._input_info
