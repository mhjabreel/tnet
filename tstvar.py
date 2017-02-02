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
v = tnet.DifferentiableVariable(x)
v.transfer('gpu')
y = T.exp(v)
y.transfer('gpu')
f = function([], y)

print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
