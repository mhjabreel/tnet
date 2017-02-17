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


__all__ = [
    "Optimizer",
]


class Optimizer(object):

    def __init__(self):
        self._defaults = {}


    """
    An abstract methdo to get the placeholders of the optimizer's parameters.
    This method shuld be implemented by the extended classes.
    """
    def _get_delegators(self):
        pass


    """
    A methdo to get the default values of the optimizer's parameters.
    """
    def _get_default(self, key):

        if key not in self._defaults:
            raise ValueError("Unknown parameter % s" % str(key))

        return self._defaults[key]

    """
    An abstract methdo to get the parameters' update function.
    This method shuld be implemented by the extended optimizers like SGD, Adadelta, ..etc.
    """

    def _get_updates(self, params, inputs):
        pass

    def define_updates(self, params):

        place_holders = self._get_delegators()
        updates = self._get_updates(params, place_holders)
        self._update_fn = theano.function(inputs=place_holders, outputs=[], updates=updates)

    def update(self, **config):

        pnames = [p.name for p in self._get_delegators()]
        inputs = []

        for p in pnames:
            if p in config:
                v = config[p]
            else:
                v = self._get_default(p)
            inputs.append(v)

        self._update_fn(*inputs)

    def __repr__(self):
        return self.__class__.__name__
