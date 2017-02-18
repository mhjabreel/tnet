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

import theano
import theano.tensor as T

import numpy as np
import math

import tnet
from tnet.nn import Module, LogSoftMax

__all__ = [
    "Criterion",
    "ClassNLLCriterion",
    "CrossEntropyCriterion",
    "BCECriterion"
]


class Criterion(object):

    def __init__(self, **kwargs):
        pass


    def __call__(self, input, target):
        return self._update_output(input, target)

    def _update_output(self, input, target):
        pass

    def forward(self, input, target):
        out = self._update_output(input, target)
        self._output = out.eval()
        return self._output


class ClassNLLCriterion(Criterion):
    def _update_output(self, input, target):
        input = T.clip(input, tnet.EPSILON, 1.0 - tnet.EPSILON)
        return T.nnet.categorical_crossentropy(input, target)


class CrossEntropyCriterion(Criterion):

    def __init__(self, **kwargs):
        self.ls = LogSoftMax()

    def _update_output(self, input, target):
        out = self.ls(input)
        return -T.mean(out[T.arange(target.shape[0]), target])


class BCECriterion(Criterion):

    def _update_output(self, input, target):

        input = T.clip(input, tnet.EPSILON, 1.0 - tnet.EPSILON)
        loss = T.mean(T.nnet.binary_crossentropy(input, target), axis=-1)

        return loss
