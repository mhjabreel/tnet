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

"""A `Type` and `Op` classes to work with numpy.ndarrays symbolically."""

import numpy
from six import integer_types
from six.moves import xrange

import theano
import theano.tensor as T
from theano.configparser import config
from theano.gof import Apply, Op
from theano.tensor import  as_tensor_variable
from theano.gradient import DisconnectedType


import logging
_logger = logging.getLogger("theano.tensor.basic")

__docformat__ = "restructuredtext en"

# This is needed as we will hide it later
python_complex = complex
python_any = any
python_all = all


__all__ = ["Linear", "linear"]

class Linear(Op):

    def make_node(self, x, W, b=None):
        """WRITEME"""

        x = as_tensor_variable(x)

        outputs = [x.type()]
        return Apply(self, [x, W, b], outputs)

    def perform(self, node, inputs, outputs):

        b = None
        if len(inputs) == 3:
            x, W, b = inputs
        elif len(inputs) == 2:
            x, W = inputs
        else:
            raise Exception("Unexpected no of inputs")

        if x.ndim == 1 or x.ndim == 2:
            y = numpy.dot(x, W)
            if not b is None:
                y += b
        else:
            raise Exception("input must be vector or matrix")

        outputs[0] = y

    def infer_shape(self, node, in_shapes):
        x_shape = in_shapes[0]
        W_shape = in_shapes[1]
        print(in_shapes)

        assert len(x_shape) in [1, 2]
        out_shape = []
        if len(x_shape) == 2:
            out_shape.append(x_shape[0])

        out_shape.append(W_shape[0])
        out_shapes = [out_shape]

        return out_shapes

    def grad(self, inputs, g_outputs):

        b = None
        gz, = g_outputs
        if len(inputs) == 3:
            x, W, b = inputs
            xdim, wdim, bdim, gdim = x.type.ndim, W.type.ndim, b.type.ndim, gz.type.ndim
            print(xdim, wdim, bdim, gdim)
        elif len(inputs) == 2:
            x, W = inputs
        else:
            raise Exception("Unexpected no of inputs")



        dy_dx = numpy.dot(gz, W)
        dy_dW = numpy.dot(gz, x)

        if not b is None:
            print(b.shape)
            one = numpy.ones_like(b)
            print(one.shape)
            dy_db = numpy.dot(gz, one.T)
            xdim, wdim, bdim, gdim = dy_dx.type.ndim, dy_dW.type.ndim, dy_db.type.ndim, gz.type.ndim
            print(xdim, wdim, bdim, gdim)
            return dy_dx, dy_dW, one

        return dy_dx, dy_dW


linear = Linear()

class SplitList(Op):
    """Partition a `TensorVariable` along some axis.
    Examples
    --------
    >>> x = matrix()
    >>> dim = 0
    You have to declare right away how many split_points there will be.
    >>> splts = split(x, dim)
    >>> f = function([x], splits)
    >>> x_inp = numpy.asarray(numpy.random.rand(5, 4), dtype=theano.config.floatX)
    >>> out = f(x_inp)
    >>> assert len(out) == x_inp.shape[dim]

    """

    dim = 0
    """A Split instance will have this many outputs, and require that
    the splits argument to `perform` have exactly this many elements.
    """
    __props__ = ("dim",)

    def __init__(self, dim=0):
        self.dim = int(dim)

    def __str__(self):
        return self.__class__.__name__ + "{%s}" % self.dim

    def make_node(self, x):
        """WRITEME"""

        x = as_tensor_variable(x)

        broadcastable = list(x.broadcastable)
        broadcastable.pop(self.dim)
        n = x.shape[self.dim]

        def _step(i):
            return [i]

        out_type = theano.tensor.TensorType(x.dtype, broadcastable)
        outputs = [out_type()]
        out_types, _ = theano.scan(_step ,sequences=[T.arange(n)], outputs_info=[])

        print(out_types)



        return Apply(self, [x], outputs)

    def perform(self, node, inputs, outputs):
        """WRITEME"""
        x, = inputs
        out, = outputs
        n = x.shape[self.dim]
        shape = list(x.shape)
        shape.pop(self.dim)


        splits = []
        sx = numpy.swapaxes(x, self.dim, 0)
        for i in xrange(n):
            s = numpy.swapaxes(sx[i], self.dim, 0).reshape(shape)
            splits.append(s)

        out[0] = splits



    def infer_shape(self, node, in_shapes):

        shp_x, = in_shapes

        out_shape = list(shp_x)
        out_shape.pop(self.dim)
        out_shapes = [out_shape]

        return out_shapes


    def grad(self, inputs, g_outputs):
        """Join the gradients along the axis that was used to split x."""
        x,  = inputs
        outputs = self(*inputs, **dict(return_list=True))
        # If all the output gradients are disconnected, then so are the inputs
        if python_all([isinstance(g.type, DisconnectedType)
                       for g in g_outputs]):
            return [DisconnectedType()()]
        # Else, we have to make them zeros before joining them
        new_g_outputs = []
        for o, g in zip(outputs, g_outputs):
            if isinstance(g.type, DisconnectedType):
                new_g_outputs.append(o.zeros_like())
            else:
                new_g_outputs.append(g)

        return [theano.tensor.concatenate(new_g_outputs, self.dim)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None for i in self.len_splits]
        return self.make_node(eval_points[0], *inputs[1:]).outputs


def split(x, dim=0):
    split_op = SplitList(dim)
    return split_op(x)
