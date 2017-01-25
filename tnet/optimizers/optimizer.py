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
import math
from tnet.base import *

__all__ = [
    "Optimizer",
    "TrainingEventArgs",
    "OnForwardEventArgs",
    "OnBackwardEventArgs"
]


class TrainingEventArgs(EventArgs):

    def __init__(self, epoch):
        super(TrainingEventArgs, self).__init__()
        self.epoch = epoch


class OnForwardEventArgs(TrainingEventArgs):

    def __init__(self, epoch, criterion_output, network_output, target):
        super(OnForwardEventArgs, self).__init__(epoch)
        self.criterion_output = criterion_output
        self.network_output = network_output
        self.target = target

class OnBackwardEventArgs(TrainingEventArgs):

    def __init__(self, epoch, params, params_grade):
        super(OnForwardEventArgs, self).__init__(epoch)
        self.params = params
        self.params_grade = params_grade
        self.target = target

class Optimizer(object):

    def __init__(self):

        self.on_start = EventHook()
        self.on_end = EventHook()

        self.on_forward = EventHook()
        self.on_backward = EventHook()

        self.on_start_poch = EventHook()
        self.on_end_epoch = EventHook()
        self.on_update = EventHook()


    def _get_train_fn(self, network, criterion):
        raise NotImplementedError

    def train(self, network, criterion, dataset_iterator, config={}):

        pass


    def test(self, network, criterion, dataset_iterator):
        pass
