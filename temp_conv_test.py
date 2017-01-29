inp = 5  #dimensionality of one sequence element
outp = 1 #number of derived features for one sequence element
kw = 1   #kernel only operates on one sequence element per step



from tnet import nn
import numpy as np

tconv = nn.TemporalConvolution(5, 10, 2)
x = np.random.rand(2, 7, 5).astype('float32') # two sequences of 7 elements
out = tconv.forward(x)
print(out.shape)
