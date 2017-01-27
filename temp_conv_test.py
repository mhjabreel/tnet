inp = 5  #dimensionality of one sequence element
outp = 1 #number of derived features for one sequence element
kw = 1   #kernel only operates on one sequence element per step



from tnet import nn
import numpy as np

tconv = nn.TemporalConvolution(inp, outp, kw)
x = np.random.rand(2, 1, inp, 7).astype('float32') # two sequences of 7 elements
out = tconv.forward(x)
print(out.shape)
