import tnet
import numpy as np
v = tnet.Parameter(np.random.randn(5,2))

x = tnet.to_shared(np.random.randn(5,2))