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


class EventArgs(object):

    def __init__(self):
        pass

class EventHook(object):
    def __init__(self):
        self.__handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self

    def invoke(self, event_args):
        for handler in self.__handlers:
            handler(event_args)



class Optimizer(object):

    def __init__(self):

        self._on_start = EventHook()
        self._on_end = EventHook()
        self._on_sample = EventHook()
        self._on_forward = EventHook()
        self._on_backward = EventHook()
        self._on_start_poch = EventHook()
        self._on_end_epoch = EventHook()
        self._on_update = EventHook()


    def train(self, network, criterion, dataset, config={}):

        pass


    def test(self, network, criterion, dataset):
        pass
