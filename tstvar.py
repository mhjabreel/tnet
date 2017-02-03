import numpy as np
import tnet
import theano
function = theano.function
T = theano.tensor
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

print(theano.config.device)
x = np.array([2.,3.]).astype(theano.config.floatX)
p = tnet.Parameter(x)
p2 = p.zero_like(None)
print(p.to_string())
