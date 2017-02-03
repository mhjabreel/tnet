
import tnet
import theano
import theano.tensor as T
import numpy as np

x = T.matrix()
n = x.shape[1]
broadcastable = list(x.broadcastable)
broadcastable.pop(1)
print(broadcastable)
out_type = T.TensorType(x.dtype, broadcastable)

y, _ = theano.scan(lambda :  [x.type()] ,n_steps=3, outputs_info=[])
print(y)
#y = tnet.split(x, 1)

f = theano.function([x], y)

x = np.random.rand(5, 3).astype(theano.config.floatX)
print(f(x))
