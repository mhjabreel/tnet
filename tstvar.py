import tnet
import numpy as np
p = tnet.Parameter(np.random.randn(5,2))
#y = p.cuda()
y = p.clone()
print(p)
print(y)
print(p.grad)
print(y.grad)
