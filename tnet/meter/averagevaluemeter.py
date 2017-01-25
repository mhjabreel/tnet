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

__all__ = ["AverageValueMeter"]

class AverageValueMeter (Meter):

    def __init__(self):
        super(AverageValueMeter, self).__init__()


    def reset(self):
        self._sum = 0
        self._weight = 0
        self._var = 0

    def add(self, value, weight=1):

        assert weight >= 0 # example weights cannot be negative

        self._sum += weight * value
        self._var += weight * value * value
        self._weight += weight

    @property
    def value(self):
        weight = self._weight
        mean = self._sum / weight
        std = math.sqrt( (self._var - weight * mean * mean) / (weight-1) )
        return mean, std
