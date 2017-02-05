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

from tnet.core import *
from tnet import nn
from tnet import dataset
from tnet import meter
from tnet import optimizers

import numpy as np
import theano
import tnet
import tnet.core
config = theano.config
to_shared = theano.shared
T = theano.tensor




device = 'cpu'
"""
def Variable(value, dtype=None, name=None):
    if not isinstance(value, np.ndarray):
        if dtype is None:
            dtype = theano.config.floatX
        value = np.asarray(value, dtype=dtype)

    var = theano.shared(value, name=name)


    return var


def Parameter(value, dtype=None, name=None):

    var = Variable(value, dtype, name)
    gname = name + "_grad" if not name is None else None
    grad = to_shared(value * 0., name=gname)
    var.grad = grad

    return var

#"""
def rand(*shape):
    x = np.random.random(shape).astype(theano.config.floatX)
    return tnet.Variable(x)


def randn(*shape):
    x = np.random.randn(*shape).astype(theano.config.floatX)
    return tnet.Variable(x)

def zeros(*shape):
    x = np.zero(shape).astype(theano.config.floatX)
    return tnet.Variable(x)


def ones(*shape):
    x = np.ones(shape).astype(theano.config.floatX)
    return tnet.Variable(x)
