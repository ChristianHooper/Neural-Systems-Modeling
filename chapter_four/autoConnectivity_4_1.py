import numpy as np, sys
import matplotlib.pyplot as plt
from vispy import app, scene
#from ch4_util import *

np.set_printoptions(threshold=sys.maxsize, linewidth=200)

# Create pattern weight matrices and subtracts diagonal mask (Inputs are in 2d arrays even if vectors)
def hebb_matrix(pattern): return pattern.T @ pattern * (1 - np.eye(len(pattern.T)))

def post_synaptic_matrix(pattern): return pattern.T @ (2 * pattern - 1) * (1 - np.eye(len(pattern.T)))

def pre_synaptic_matrix(pattern): return ( 2 * pattern.T - 1) @ pattern * (1 - np.eye(len(pattern.T))) # //

def Hopfield_matrix(pattern): 
    #print(f'HOP{( 2 * pattern.T - 1)} @ {(2 * pattern - 1)}')
    return ( 2 * pattern.T - 1) @ (2 * pattern - 1) * (1 - np.eye(len(pattern.T))) # Covariance matrix

#  /////////////////////////////////////////////////////
#TODO: Async implemented, but not falling into attract pattern
def asymmetric_sequence(pattern, bias: int=2): #return np.array([[( 2 * pattern[n] - 1) @ (2 * pattern.T[n] - 1)] for n in range(-1, len(pattern)-1)])
    asymmetrical = np.zeros((len(pattern[0]), len(pattern[0])))
    base_pattern = pattern
    pattern = 2 * pattern - 1
    #print(asymmetrical)
    #print(pattern)
    #print(np.roll(pattern,1,axis=0),'\n\n')
    #print(f'{np.roll(pattern,1,axis=0).T} @ {pattern}')
    #print(f'MATRICES: {np.roll(pattern,1,axis=0).T @ pattern}')
    '''
    for n in range(-1, len(pattern)-1):
        #print(f'{n+1}@{n}')

        asymmetrical += np.outer(pattern[n+1], pattern[n].T)
    '''
    asymmetrical = np.roll(pattern,-1,axis=0).T @ pattern
    print('ASM:\n', (asymmetrical * bias)  * (1 - np.eye(len(pattern.T))))
    return (asymmetrical * bias) * (1 - np.eye(len(pattern.T)))

ex = np.array([[1,0,1,0],[1,1,0,0],[0,1,1,0]])

#print(asymmetric_sequence(ex))
#print(Hopfield_matrix(ex))
#  /////////////////////////////////////////////////////


def seq_weights(patterns, bias=0.2, loop=True):
    # patterns: (P,N) in 0/1
    X = 2*patterns - 1   # to ±1
    N = X.shape[1]

    # Symmetric Hopfield part
    W_sym = (X.T @ X) / N
    np.fill_diagonal(W_sym, 0.0)

    # Asymmetric part: next <- current
    if loop:
        X_next = np.roll(X, -1, axis=0)   # shift forward
    else:
        X_next = np.vstack([X[1:], np.zeros_like(X[0])])
    W_asym = (X_next.T @ X) / N
    np.fill_diagonal(W_asym, 0.0)

    return (1 - bias)*W_sym + bias*W_asym


# /////////////////////////////////////////////////////

rng = np.random.default_rng()
one = np.zeros(20); one[1:20:2] = 1
two = np.zeros(20); one[1:20:5] = 1
three=np.zeros(20); three[1:4]=1; three[5:8]=1; three[9:12]=1; three[13:16]=1; three[17:20]=1
y_iii = np.array([ # rng.integers(0,2, size=(3,20), dtype=int)
    one,
    two,
    three
    ])

#Hopfield_matrix(y_iii)
#print(asymmetric_sequence(y_iii))
initial_pattern = rng.integers(0,2, size=(1,20), dtype=int)
#print('Pattern:\n', y_iii)
#print('initial_pattern: ', initial_pattern)
async_order = np.random.choice(np.arange(0,len(initial_pattern), dtype=int), size=len(initial_pattern), replace=False)


#ex = np.array([[1,1,0], [0,0,1]])

'''
print(hebb_matrix(q),'\n')
print(post_synaptic_matrix(q),'\n')
print(pre_synaptic_matrix(q),'\n')
print(Hopfield_matrix(q))
'''

time_step: int = 200
y = np.zeros((time_step, len(y_iii[0]))) # Neuron state time series
y[0] = initial_pattern # Initial neural state
w = asymmetric_sequence(y_iii)
#print(f'Weights:\n{w}')
'''
# Basic synchronous network equation: $q(t)=W\cdot{y(t-1)}$
for t in range(1, time_step):
    y[t] = (w @ y[t-1] > 0).astype(int) # Remaps neural dot-product output to boolean array then converts to binary
#print(f'Sync:\n {y}\n')p
y = np.zeros((time_step, len(y_ii[0]))) # Neuron state time series
y[0] = initial_pattern # Initial neural state
'''
#print(w,'\n')
# Basic asynchronous network equation: $q_i(t)=W_{row,i}\cdot{y(t-1)}$
for t in range(1, time_step):
    y_copy = y[t-1].copy()
    async_order = rng.permutation(len(y_copy))

    for i in async_order:
        q_i = w[i] @ y_copy # Remaps neural dot-product
        y_copy[i] = y_copy[i] if q_i == 0 else (1 if q_i > 0 else 0)

    y[t] = y_copy
print(f'A-sync:\n {y}\n')

# Rendering
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = scene.cameras.PanZoomCamera(rect=(0, 0, 2, 10))

'''
# Create scatter plot for markers
scatter = scene.visuals.Markers()
view.add(scatter)

# Initial data
pos = np.column_stack((x_vals, y_vals))
scatter.set_data(pos, face_color='red', size=10)

# Update function
step = 0
def update(ev):
    scatter.set_data(pos, face_color='red', size=10)

# Timer to animate
timer = app.Timer(interval=0.05, connect=update, start=True)

if __name__ == '__main__':
    app.run()
'''