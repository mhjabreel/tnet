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

import tnet
import theano
import theano.tensor.basic
from theano.sandbox.cuda.type import CudaNdarrayType
import theano.sandbox.cuda
from theano.sandbox.cuda import use, unuse, CudaNdarray, filter as type_support_filter
#from tnet.core.variable import Variable as CPU_Variable, Parameter as CPU_Parameter
#from tnet.cuda.variable import Variable as GPU_Variable, Parameter as GPU_Parameter

# We can't test the driver during import of theano.sandbox.cuda as
# this cause circular import dependency. So we also test it manually
# after the import

if theano.sandbox.cuda.cuda_available:
    import theano.sandbox.cuda.tests.test_driver


    theano.sandbox.cuda.tests.test_driver.test_nvidia_driver1()


@staticmethod
def get_cuda_type(value):
    assert isinstance(value, np.ndarray), "Expected numpy.ndarray value got %s" % value
    broadcastable = (False,) * len(value.shape)
    type = CudaNdarrayType(broadcastable=broadcastable)
    get_value_return_ndarray = True
    if isinstance(value, CudaNdarray):
        deviceval = value.copy()
    else:
        # type.broadcastable is guaranteed to be a tuple, which this next
        # function requires
        deviceval = type_support_filter(value, type.broadcastable, False, None)
    return type, deviceval

def device(d):
    if type(d) == int:
        if d < 0:
            d = 'cpu'

            #tnet.Variable = CPU_Variable#get_type = tnet.get_tensor_type#
            #tnet.Parameter = CPU_Parameter
            unuse()
            tnet.device = 'cpu'
            return
        else:
            d = 'gpu' +  str(d)
    assert type(d) == str
    use(d, force=True, default_to_move_computation_to_gpu=True, move_shared_float32_to_gpu=False, test_driver=False)

    #tnet.Variable = GPU_Variable#get_type = tnet.get_tensor_type#
    #tnet.Parameter = GPU_Parameter
    tnet.device = d
