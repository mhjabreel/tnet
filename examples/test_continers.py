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

print("Test ParallelList")

m = nn.ParallelList()
m.add(nn.Linear(10, 2))
m.add(nn.Linear(5, 3))

x = tnet.randn(10)
y = tnet.rand(5)

out = m.forward([x, y])
for i, k in enumerate(out):
    print(i)
    print(k)

print("Test ConcatList")

m = nn.ConcatList()
m.add(nn.Linear(5, 2))
m.add(nn.Linear(5, 3))

x = tnet.rand(5)

out = m.forward(x)
for i, k in enumerate(out):
    print(i)
    print(k)


print("Test ConcatList With JoinList")


m = nn.ConcatList()
m.add(nn.Linear(5, 2))
m.add(nn.Linear(5, 2))

m = nn.Sequential()\
    .add(m) \
    .add(nn.JoinList(0))
x = tnet.rand(5)

out = m.forward(x)
print(out)


print("CAddList")
ii = [tnet.ones(5), tnet.ones(5) * 2, tnet.ones(5) * 3]
m = nn.CAddList()
out = m.forward(ii)
print(out)

print("CSubList")
ii = [tnet.ones(5) * 2.2, tnet.ones(5) ]
m = nn.CSubList()
out = m.forward(ii)
print(out)

print("CMulList")
ii = [tnet.ones(5, 2) * 2, tnet.ones(5, 2) * 3,   tnet.ones(5, 2) * 4]
m = nn.CMulList()
out = m.forward(ii)
print(out)

print("CDivList")
ii = [tnet.ones(5) * 2.2, tnet.ones(5) * 4.4]
m = nn.CDivList()
out = m.forward(ii)
print(out)

print("CDivList2")
ii = [tnet.ones(5, 2) * 2.2, tnet.ones(5, 2) * 4.4]
m = nn.CDivList()
out = m.forward(ii)
print(out)

print("CMaxList")
ii = [tnet.Variable(np.array([1, 2, 3])), tnet.Variable(np.array([3, 2, 1]))]
m = nn.CMaxList()
out = m.forward(ii)
print(out)

print("CMaxList2")
ii = [tnet.Variable(np.array([[1, 2, 3], [3, 2, 1]])), tnet.Variable(np.array([[3, 2, 1], [1, 2, 3]]))]
m = nn.CMaxList()
out = m.forward(ii)
print(out)

print("CMinList")
ii = [tnet.Variable(np.array([1, 2, 3])), tnet.Variable(np.array([3, 2, 1]))]
m = nn.CMinList()
out = m.forward(ii)
print(out)

print("CMinList2")
ii = [tnet.Variable(np.array([[1, 2, 3], [3, 2, 1]])), tnet.Variable(np.array([[3, 2, 1], [1, 2, 3]]))]
m = nn.CMinList()
out = m.forward(ii)
print(out)
