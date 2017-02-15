import tnet
import theano
import theano.tensor as T
import numpy as np

print("Running on: " + tnet.device)
print(tnet.Variable)
p = tnet.Variable(np.random.randn(5,2).astype('float32'))
print(p)

y = p + 2
f = theano.function([], y)
print(f.maker.fgraph.toposort())
if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


import tnet.cuda as cuda
cuda.device('gpu')
print("Running on: " + tnet.device)
p = tnet.Variable(np.random.randn(5,2).astype('float32'))
print(p)
y = p + 2
f = theano.function([], y)
print(f.maker.fgraph.toposort())
if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
