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


from tnet.meter import Meter
import numpy as np
import math

__all__ = ["AccuracyMeter", "BinaryAccuracyMeter"]

class BinaryAccuracyMeter(Meter):
    def __init__(self, error=False):
        super(BinaryAccuracyMeter, self).__init__()
        self._error = error


    def reset(self):
        self._sum = 0
        self._count = 0

    def add(self, output, target):



        assert type(output) == type(target) == np.ndarray
        output = np.round(output)
        score = (target == output)

        self._sum += score.sum()
        self._count += len(score)

    @property
    def value(self):
        acc = self._sum / self._count * 100
        if self._error:
            return 100 - acc
        return acc

class AccuracyMeter (Meter):

    def __init__(self, error=False):
        super(AccuracyMeter, self).__init__()
        self._error = error



    def reset(self):
        self._sum = 0
        self._count = 0

    def add(self, output, target):



        assert type(output) == type(target) == np.ndarray


        if target.ndim == 2:
            assert target.max() == 1 and target.min() == 0 # one hot check
            target = np.argmax(target, axis=1)

        if output.ndim == 2:
            output = np.argmax(output, axis=1)


        score = (target == output)

        self._sum += score.sum()
        self._count += len(score)

    @property
    def value(self):
        acc = self._sum / self._count * 100
        if self._error:
            return 100 - acc
        return acc
