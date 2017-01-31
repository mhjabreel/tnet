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
import tnet
from tnet import nn

print("Replicate example")
x = tnet.rand(5, 2)
m = nn.Replicate(3, 1)
o = m.forward(x)
print(o)
print(o.shape)
print("Replicate with ndim example")
m = nn.Replicate(3, 0, 1)
o = m.forward(x)
print(o)
print(o.shape)

print("Narrow example")
x = tnet.rand(4, 5)
m = nn.Narrow(1, 2, 3)
print(x)
print(o)
print(o.shape)


print("Select example")
m = tnet.nn.Sequential()
m.add(tnet.nn.Select(0, 3))
x = tnet.rand(10, 5)
o = m.forward(x)
print(x)
print(o)
print(o.shape)


print("Squeeze example")
m = tnet.nn.Sequential()
m.add(tnet.nn.Squeeze(1))
x = tnet.rand(5,1,2)
o = m.forward(x)
print(x)
print(o)
print(o.shape)

print("Squeeze with ndim example")
m = tnet.nn.Sequential()
m.add(tnet.nn.Squeeze(0, 2))
x = tnet.rand(5,1,2)
o = m.forward(x)
print(x)
print(o)
print(o.shape)

print("Unsqueeze example")
m = tnet.nn.Sequential()
m.add(tnet.nn.Unsqueeze(1))
x = tnet.rand(5,2)
o = m.forward(x)
print(x)
print(o)
print(o.shape)

print("Unsqueeze with ndim example")
m = tnet.nn.Sequential()
m.add(tnet.nn.Unsqueeze(0, 2))
x = tnet.rand(5,2)
o = m.forward(x)
print(x)
print(o)
print(o.shape)

print("Clamp example")
m = tnet.nn.Sequential()
m.add(tnet.nn.Clamp(-0.1, 0.5))
x = tnet.randn(2, 5)
o = m.forward(x)
print(x)
print(o)
print(o.shape)

print("SplitList example")
m = nn.Sequential()
m.add(nn.SplitList(0))
x = tnet.randn(2, 5)
o = m.forward(x)
print(x)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(len(o))
print(o)
print(type(o))
print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

m = nn.JoinList(0)
o = m.forward(o)
print(o)
print(o.shape)

print("SplitList with ndim example")
m = nn.Sequential()
m.add(nn.SplitList(0, 1))
x = tnet.randn(2, 5)
o = m.forward(x)
print(x)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(len(o))
print(o)
