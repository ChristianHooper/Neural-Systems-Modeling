import numpy as np
import matplotlib.pyplot as plt
from ch4_util import *

p = np.array(
    [[1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1]]
    )

ex = np.array([[1,1,0], [0,0,1]])

#print(hebb_matrix(p))
#print(post_synaptic_matrix(p))
#print(pre_synaptic_matrix(p))
print(Hopfield_matrix(ex))

neural_state = np.array([.4, .3, .5, .3, .4, .2, .1, .1, .2, .1])
