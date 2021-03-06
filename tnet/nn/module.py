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
from theano.tensor.sharedvar import TensorSharedVariable
from theano.sandbox.cuda.var import CudaNdarraySharedVariable

from copy import deepcopy
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
        self._params = []
        self._declare(**kwargs)

    #@staticmethod
    def _check_input(self, input_or_inputs):

        type_of_input = type(input_or_inputs)

        if type_of_input == list or type_of_input == tuple:

            type_of_inputs = [type(inp) for inp in input_or_inputs]

            all_types_are_correct = all([t == np.ndarray or
                                          t == T.TensorVariable or
                                          t == tnet.Variable or
                                          t == tnet.Parameter or
                                          t == T.TensorConstant
                                          for t in type_of_inputs])

            if not all_types_are_correct:
                raise ValueError("Wrong types are passed %s" %inp)

            input_or_inputs = [tnet.Variable(inp) if isinstance(t, np.ndarray) else
                               inp for inp in input_or_inputs]
        elif type_of_input == np.ndarray:

            input_or_inputs = to_tensor(input_or_inputs)#tnet.Variable(input_or_inputs)
        else:
            if not (type_of_input == T.TensorVariable or
                            type_of_input == T.TensorConstant or
                            type_of_input == tnet.Variable or
                            type_of_input == tnet.Parameter):
                raise  ValueError("Wrong types are passed")

        return input_or_inputs

    def _update_output(self, input_or_inputs):

        input_or_inputs = self._check_input(input_or_inputs)
        self.input = input_or_inputs
        return input_or_inputs

    def __call__(self, input_or_inputs):
        return self._update_output(input_or_inputs)

    def forward(self, input_or_inputs):

        out = self._update_output(input_or_inputs)
        if type(out) == list:
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
        return self._params


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

    def __repr__(self):

        return self.__class__.__name__

    def __str__(self):
        return str(self.__repr__())

    def clone(self):
        return deepcopy(self)

    def share(self):
        pass
