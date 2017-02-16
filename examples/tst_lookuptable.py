

from tnet import nn
import numpy as np
import theano
T = theano.tensor

x = T.vector()

lkp = nn.LookupTable(10, 5)

y = lkp(x)
f = theano.function([x], y)
x = np.random.randint(0, 10, size=(5))
out = f(x)#lkp.forward(x)
print(out)
print(out.shape)

m = nn.View(-1, 25)
out = m.forward(out)#lkp.forward(x)
print(out)
print(out.shape)
