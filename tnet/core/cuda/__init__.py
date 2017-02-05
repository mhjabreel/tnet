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
import copy
import numpy as np


import theano
import theano.tensor.basic

import theano.sandbox.cuda
from theano.sandbox.cuda import use
# We can't test the driver during import of theano.sandbox.cuda as
# this cause circular import dependency. So we also test it manually
# after the import
if theano.sandbox.cuda.cuda_available:
    import theano.sandbox.cuda.tests.test_driver

    if config.enable_initial_driver_test:
        theano.sandbox.cuda.tests.test_driver.test_nvidia_driver1()


def device(d):
    if type(d) == int:
        if d < 0:
            d = 'cpu'
        else:
            d = gpu +  str(d)

    assert type(d) == string
    use(d, force=config.force_device, default_to_move_computation_to_gpu=False,move_shared_float32_to_gpu=False, test_driver=False)
