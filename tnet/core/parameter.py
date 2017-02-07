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
import numpy as np
from tnet import Variable


class Parameter(Variable):
    """docstring for Parameter."""
    def __init__(self, value, name=None, type=None):
        super(Parameter, self).__init__(value, name, type)
        gname = name + "_grad" if not name is None else None
        self.grad = Variable(np.array(self.container.value), gname)
        self.grad.zero()