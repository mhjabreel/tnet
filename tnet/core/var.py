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

class _var(object):
    def __repr__(self):
        value = self.container.value
        t = str(value.dtype)
        t = t[0].upper() + t[1:]
        size = value.shape
        value = str(value)
        return value + "\n" + str(t) + "Tensor of size " + str(size)

    def __str__(self):

        return self.__repr__()

    @property
    def data(self):
        return self.get_value()
