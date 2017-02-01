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
import theano
import math

import tnet
from tnet.nn import Module, Container, InputInfo

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config

class ParallelList(Container):
    """
    ParallelList is a container module that, in its forward() method, applies the i-th member module to the i-th input, and outputs a table of the set of outputs.

    +----------+         +-----------+
    | {input1, +---------> {member1, |
    |          |         |           |
    |  input2, +--------->  member2, |
    |          |         |           |
    |  input3} +--------->  member3} |
    +----------+         +-----------+
    Example

    >>> m = nn.Parallel()
    >>> m.add(nn.Linear(10, 2))
    >>> m.add(nn.Linear(5, 3))

    >>> x = tnet.randn(10)
    >>> y = tnet.rand(5)

    >>> out = m.forward([x, y])
    >>> for i, k in enumerate(out):
        ... print(i)
        ... print(k)
    which gives the output:

    0
    [ 0.52960086  0.34379268]
    float32_tensor of size (2,)
    1
    [-0.09274504  0.58627141  0.87921637]
    float32_tensor of size (3,)
    """
    def __init__(self):
        super(ParallelList, self).__init__()


    def _update_output(self, inp):

        outputs = []
        assert type(inp) == list
        inp = self._check_input(inp)

        for i, m in enumerate(self._modules):
            outputs.append(m(inp[i]))

        return outputs


    @property
    def input_info(self):
        return [m.input_info for m in self._modules]


class ConcatList(Container):
    """
    ConcatList is a container module that applies each member module to the same input.

                      +-----------+
                 +----> {member1, |
    +-------+    |    |           |
    | input +----+---->  member2, |
    +-------+    |    |           |
                 +---->  member3} |
                      +-----------+

    Example 1:

    >>> m = nn.ConcatList()
    >>> m.add(nn.Linear(5, 2))
    >>> m.add(nn.Linear(5, 3))

    >>> x = tnet.randn(5)

    >>> out = m.forward(x)
    >>> for i, k in enumerate(out):
        ... print(i)
        ... print(k)
    which gives the output:

    0
    [-0.17492479 -0.23794149]
    float32_tensor of size (2,)
    1
    [-0.01698744  1.62631071  0.6912719 ]
    float32_tensor of size (3,)

    Example 2:

    >>> m = nn.ConcatList()
    >>> m.add(nn.Linear(5, 2))
    >>> m.add(nn.Linear(5, 2))

    >>> m = nn.Sequential().add(m).add(nn.JoinList(0))
    >>> x = tnet.rand(5)

    >>> out = m.forward(x)
    >>> print(out)

    which gives the output:
    [[-0.10930242  0.69323033]
     [ 0.86695296  0.65297145]]
    float32_tensor of size (2, 2)
    """

    def __init__(self):
        super(ConcatList, self).__init__()

    def _update_output(self, inp):

        outputs = []

        inp = self._check_input(inp)

        for i, m in enumerate(self._modules):
            outputs.append(m(inp))

        return outputs


    @property
    def input_info(self):
        if len(self._modules) > 0:
            m = self._modules[0]
            return m.input_info


class SplitList(Module):
    """
    Creates a module that takes a Variable as input and outputs list of variables,
        splitting the variable along the specified dimension. In the diagram below, dimension is equal to 1.

        +----------+         +-----------+
        | input[1] +---------> {member1, |
     +----------+-+          |           |
      | input[2] +----------->  member2, |
    +----------+-+           |           |
    | input[3] +------------->  member3} |
    +----------+             +-----------+
    The optional parameter nInputDims allows to specify the number of dimensions that this module will receive.
    This makes it possible to forward both minibatch and non-minibatch Tensors through the same module.
    """

    def __init__(self, nb_splits, dim, ndim=None):
        self._dim = dim
        self._ndim = ndim
        self._nb_splits = nb_splits
        super(SplitList, self).__init__()


    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):
        inp = self._check_input(inp)
        dims = self._nb_splits
        splist_sizes = [1] * dims
        dim = self._dim + 1 if not self._ndim is None else self._dim
        splits = T.split(inp, splist_sizes, dims, dim)
        return splits



class JoinList(Module):
    """docstring for JoinList."""
    def __init__(self, dim, ndim=None):
        self._dim = dim
        self._ndim = ndim
        super(JoinList, self).__init__()

    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):
        dim = self._dim + 1 if not self._ndim is None else self._dim
        inp = self._check_input(inp)

        return T.concatenate([inp], dim)


class CAddList(Module):
    """
    Takes a list of Tensors and outputs summation of all Tensors.

    Example:
    >>> ii = [tnet.ones(5), tnet.ones(5) * 2, tnet.ones(5) * 3]
    >>> m = nn.CAddList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:
    [ 6.  6.  6.  6.  6.]
    float32_tensor of size (5,)

    """
    def __init__(self):
        super(CAddList, self).__init__()

    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):

        assert type(inp) == list and len(inp) > 0
        inp = self._check_input(inp)

        s = inp[0]
        for i in range(1, len(inp)):
            s += inp[i]

        return s

class CSubList(Module):
    """
    Takes a list of Tensors and outputs component-wise subtraction between them.

    Example:
    >>> ii = [tnet.ones(5) * 2.2, tnet.ones(5) ]
    >>> m = nn.CSubList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:
    [ 1.20000005  1.20000005  1.20000005  1.20000005  1.20000005]
    float32_tensor of size (5,)

    """
    def __init__(self):
        super(CSubList, self).__init__()

    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):

        assert type(inp) == list and len(inp) > 0
        inp = self._check_input(inp)

        s = inp[0]
        for i in range(1, len(inp)):
            s -= inp[i]

        return s

class CMulList(Module):
    """
    Takes a list of Tensors and outputs the multiplication of all of them.

    Example:
    >>> ii = [tnet.ones(5, 2) * 2, tnet.ones(5, 2) * 3,   tnet.ones(5, 2) * 4]
    >>> m = nn.CMulList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:
    [[ 24.  24.]
     [ 24.  24.]
     [ 24.  24.]
     [ 24.  24.]
     [ 24.  24.]]
    float32_tensor of size (5, 2)
    """
    def __init__(self):
        super(CMulList, self).__init__()

    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):

        assert type(inp) == list and len(inp) > 0
        inp = self._check_input(inp)
        return T.prod(inp, 0)


class CDivList(Module):
    """
    Takes a list of Tensors and returns the component-wise division between them.

    Example 1:
    >>> ii = [tnet.ones(5) * 2.2, tnet.ones(5) * 4.4]
    >>> m = nn.CDivList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:
    [ 0.5  0.5  0.5  0.5  0.5]
    float32_tensor of size (5,)

    Example 2:
    >>> ii = [tnet.ones(5, 2) * 2.2, tnet.ones(5, 2) * 4.4]
    >>> m = nn.CDivList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:

    [[ 0.5  0.5]
     [ 0.5  0.5]
     [ 0.5  0.5]
     [ 0.5  0.5]
     [ 0.5  0.5]]
    float32_tensor of size (5, 2)
    """
    def __init__(self):
        super(CDivList, self).__init__()

    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):

        assert type(inp) == list and len(inp) > 0
        inp = self._check_input(inp)

        s = inp[0]
        for i in range(1, len(inp)):
            s /= inp[i]
        return s


class CMaxList(Module):
    """
    Takes a list of Tensors and outputs the max of all of them.

    Example 1:
    >>> ii = [tnet.Variable(np.array([1, 2, 3])), tnet.Variable(np.array([3, 2, 1]))]
    >>> m = nn.CMaxList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:
    [3 2 3]
    int64_tensor of size (3,)

    Example 2:
    >>> ii = [tnet.Variable(np.array([[1, 2, 3], [3, 2, 1]])), tnet.Variable(np.array([[3, 2, 1], [1, 2, 3]]))]
    >>> m = nn.CMaxList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:

    [[3 2 3]
     [3 2 3]]
    int64_tensor of size (2, 3)

    """
    def __init__(self):
        super(CMaxList, self).__init__()

    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):

        assert type(inp) == list and len(inp) > 0
        inp = self._check_input(inp)
        return T.max(inp, 0)

class CMinList(Module):
    """
    Takes a list of Tensors and outputs the min of all of them.

    Example 1:
    >>> ii = [tnet.Variable(np.array([1, 2, 3])), tnet.Variable(np.array([3, 2, 1]))]
    >>> m = nn.CMinList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:
    [1 2 1]
    int64_tensor of size (3,)

    Example 2:
    >>> ii = [tnet.Variable(np.array([[1, 2, 3], [3, 2, 1]])), tnet.Variable(np.array([[3, 2, 1], [1, 2, 3]]))]
    >>> m = nn.CMinList()
    >>> out = m.forward(ii)
    >>> print(out)

    output:

    [[1 2 1]
     [1 2 1]]
    int64_tensor of size (2, 3)

    """
    def __init__(self):
        super(CMinList, self).__init__()

    def _declare(self):
        pass

    def _compile(self):
        pass

    def _update_output(self, inp):

        assert type(inp) == list and len(inp) > 0
        inp = self._check_input(inp)
        return T.min(inp, 0)
