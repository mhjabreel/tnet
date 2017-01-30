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

from tnet.nn import Module, InputInfo

T  = theano.tensor
func = theano.function
to_tensor = T.as_tensor_variable
to_shared = theano.shared
config = theano.config


class _GobalPooling(Module):
    def __init__(self, pool_fn, dimension, n_input_dim=None):


        self._dimension = dimension
        self._n_input_dim = n_input_dim
        self._pool_fn = pool_fn
        super(_GobalPooling, self).__init__()


    def _get_positive_index(self, inp):
        d = self._dimension
        if  d < 0:
            d = inp.ndim + d + 1
        elif not self._n_input_dim is None and inp.ndim == (self._n_input_dim + 1):
            d = d + 1
        return d

    def _declare(self):
        pass


    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)
        d = self._get_positive_index(inp)
        return self._pool_fn(inp, axis=d)


class Max(_GobalPooling):

    """
    Applies a max operation over dimension dimension.
    The parameters are the following:

    Hence, if an nxpxq Tensor was given as input, and dimension = 2 then an nxq matrix would be output.
    When n_input_dim is provided, inputs larger than that value will be considered batches
        where the actual dimension to apply the max operation will be dimension dimension + 1.
    """

    def __init__(self, dimension, n_input_dim=None):

        super(Max, self).__init__(T.max, dimension, n_input_dim)


class Min(_GobalPooling):

    """
    Applies a min operation over dimension dimension.
    The parameters are the following:

    Hence, if an nxpxq Tensor was given as input, and dimension = 2 then an nxq matrix would be output.
    When n_input_dim is provided, inputs larger than that value will be considered batches
        where the actual dimension to apply the max operation will be dimension dimension + 1.
    """

    def __init__(self, dimension, n_input_dim=None):

        super(Min, self).__init__(T.min, dimension, n_input_dim)

class Mean(_GobalPooling):

    """
    Applies a mean operation over dimension dimension.
    The parameters are the following:

    Hence, if an nxpxq Tensor was given as input, and dimension = 2 then an nxq matrix would be output.
    When n_input_dim is provided, inputs larger than that value will be considered batches
        where the actual dimension to apply the max operation will be dimension dimension + 1.
    """

    def __init__(self, dimension, n_input_dim=None):

        super(Mean, self).__init__(T.mean, dimension, n_input_dim)


class Sum(_GobalPooling):

    """
    Applies a max operation over dimension dimension.
    The parameters are the following:

    Hence, if an nxpxq Tensor was given as input, and dimension = 2 then an nxq matrix would be output.
    When n_input_dim is provided, inputs larger than that value will be considered batches
        where the actual dimension to apply the max operation will be dimension dimension + 1.
    """

    def __init__(self, dimension, n_input_dim=None):

        super(Sum, self).__init__(T.sum, dimension, n_input_dim)

class Flatten(Module):

    def __init__(self):

        super(Flatten, self).__init__()


    def _declare(self):
        pass


    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)

        y = T.reshape(inp, (inp.shape[0], T.prod(inp.shape) // inp.shape[0]))

        return y


class Transpose(Module):

    def __init__(self, *dargs):
        assert len(dargs) > 0 #and len(*dargs) <= 4
        assert all([type(v) == int for v in dargs])
        assert min(*dargs) == 0
        self._view_pattern = dargs#[v if v >= 0 else 'x' for v in dargs]
        super(Transpose, self).__init__()

    def _compile(self):
        pass
    def _declare(self):
        pass

    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)
        return inp.dimshuffle(self._view_pattern)

class View(Module):

    def __init__(self, *dargs):
        assert len(dargs) > 0 #and len(*dargs) <= 4
        assert all([type(v) == int for v in dargs])
        assert min(*dargs) == -1
        c = 0
        s = 0
        negidx = None
        for i, v in enumerate(dargs):
            if v == -1:
                c += 1
                negidx = i
            else:
                s += v
            if c > 1:
                raise ValueError("only one dimension can be at -1")
        self._new_shape = list(dargs)#[v if v >= 0 else 'x' for v in dargs]
        self._non_negative_dim = s
        self._negidx = negidx
        super(View, self).__init__()

    def _get_shape(self, inp):
        shape = self._new_shape
        return tuple(shape)

    def _compile(self):

        shape = self._new_shape
        if not self._negidx is None:
            shape[self._negidx] = 1
        self.forward(np.random.random(shape).astype(config.floatX))

    def _declare(self):
        pass

    def _update_output(self, inp):
        inp = self._prpare_inputs(inp)
        assert isinstance(inp, T.TensorConstant) or isinstance(inp, T.TensorVariable)
        shape = self._get_shape(inp)
        return T.reshape(inp, shape)


class Replicate(Module):

    """
    This class creates an output where the input is replicated nfeature times along dimension dim (default 0).
    There is no memory allocation or memory copy in this module. It sets the stride along the dimth dimension to zero.
    When provided, ndim should specify the number of non-batch dimensions.
    This allows the module to replicate the same non-batch dimension dim for both batch and non-batch inputs.
    """
    def __init__(self, nfeature, dim=0, ndim=None):
        self._nfeature = nfeature
        self._dim = dim
        self._ndim = ndim

        super(Replicate, self).__init__()


class Narrow(Module):
    """
    Narrow is application of narrow operation in a module.
    The module further supports negative length, dim and offset to handle inputs of unknown size.
    """

    pass


class Select(Module):

    """
    Selects a dimension and index of a nxpxqx.. Variable.
    example:

    >> m = tnet.nn.Sequential()
    >> m.add(tnet.nn.Select(1, 3))

    >> x = tnet.randn(10, 5)
    >> print(x)
    >> print(m.forward(x))
    gives the output:

     [[0.9720 -0.0836  0.0831 -0.2059 -0.0871],
      [0.8750 -2.0432 -0.1295 -2.3932  0.8168]
      ........................................
     [0.5804 -0.5333  1.1621  1.5683 -0.1978]]
    tnet.float32_variable of size (10, 5)

     [0.0369
     1.1633
     0.6483
     1.2862
     0.6596]
    tnet.float32_variable of size (5,)
    """
    pass

class Squeeze(Module):
    """
    Applies the Variable squeeze operation.
    """
    def __init__(self, dim, numInputDims):
        super(Squeeze, self).__init__()
        pass

class Unsqueeze(Module):
    """
    Insert singleton dim (i.e., dimension 1) at position pos.
    For an input with dim = input:dim(), there are dim + 1 possible positions to insert the singleton dimension.
    For example, if input is 3 dimensional Tensor in size p x q x r,
    then the singleton dim can be inserted at the following 4 positions
    pos = 1: 1 x p x q x r
    pos = 2: p x 1 x q x r
    pos = 3: p x q x 1 x r
    pos = 4: p x q x r x 1

    Indicate the expected input feature map dimension by specifying numInputDims.
    This allows the module to work with mini-batch.
    """
    def __init__(self, pos , numInputDims):
        super(Unsqueeze, self).__init__()
        self.arg = arg

class Exp(Module):
    """
    Applies the exp function element-wise to the input Tensor, thus outputting a Tensor of the same dimension.
    """
    def __init__(self, arg):
        super(Exp, self).__init__()
        self.arg = arg

class Log(Module):
    """Applies the log function element-wise to the input Tensor, thus outputting a Tensor of the same dimension."""
    def __init__(self, arg):
        super(Log, self).__init__()
        self.arg = arg

class Square(Module):
    """Takes the square of each element."""
    def __init__(self):
        super(Square, self).__init__()


class Sqrt(Module):
    """Takes the square root of each element."""
    def __init__(self):
        super(Sqrt, self).__init__()


class Power(Module):
    """Raises each element to its p-th power."""
    def __init__(self, p):
        super(Power, self).__init__()
        self.arg = arg

class Clamp(Module):
    """
    Clamps all elements into the range [min_value, max_value].
    Output is identical to input in the range,
    otherwise elements less than min_value (or greater than max_value) are saturated to min_value (or max_value).
    """
    def __init__(self, min_value, max_value):
        super(Clamp, self).__init__()
        self.arg = arg

class Normalize(Module):
    """
    Normalizes the input Tensor to have unit L_p norm.
    The smoothing parameter eps prevents division by zero when the input contains all zero elements (default = 1e-10).

    Input can be 1D or 2D (in which case it's considered as in batch mode)
    """
    def __init__(self, p, eps):
        super(Normalize, self).__init__()
        self.arg = arg

class MM(Module):
    """
    Performs multiplications on one or more pairs of matrices.
    If transA is set to true, the first matrix is transposed before multiplication.
    If transB is set to true, the second matrix is transposed before multiplication.
    By default, the matrices do not get transposed.

    The module also accepts 3D inputs which are interpreted as batches of matrices.
    When using batches, the first input matrix should be of size b x m x n
        and the second input matrix should be of size b x n x p (assuming transA and transB are not set).
    If transA or transB is set, transpose takes place between the second and
        the third dimensions for the corresponding matrix.
    """
    def __init__(self, transA, transB):
        super(MM, self).__init__()
        self.arg = arg

class BatchNormalization(Module):
    """
    where N is the dimensionality of input eps is a small value added to the standard-deviation to avoid divide-by-zero.
    Defaults to 1e-5. affine is a boolean. When set to false, the learnable affine transform is disabled. Defaults to true.
    During training, this layer keeps a running estimate of its computed mean and std.
    The running sum is kept with a default momentum of 0.1 (unless over-ridden) During evaluation,
    this running mean/std is used for normalization.

    Implements Batch Normalization as described in the paper:
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    by Sergey Ioffe, Christian Szegedy.

    The operation implemented is:

                  x - mean(x)
    y =  ----------------------------- * gamma + beta
          standard-deviation(x) + eps
    where the mean and standard-deviation are calculated per-dimension over the mini-batches and
    where gamma and beta are learnable parameter vectors of size N (where N is the input size).
    The learning of gamma and beta is optional. The module only accepts 2D inputs.
    """
    def __init__(self, arg):
        super(BatchNormalization, self).__init__()
        self.arg = arg

class Padding(Module):
    """docstring for Padding."""
    def __init__(self, arg):
        super(Padding, self).__init__()
        self.arg = arg


class L1Penalty(Module):
    """docstring for L1Penalty."""
    def __init__(self, arg):
        super(L1Penalty, self).__init__()
        self.arg = arg
