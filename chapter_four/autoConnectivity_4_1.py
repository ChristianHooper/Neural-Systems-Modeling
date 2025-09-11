import numpy as np
import matplotlib.pyplot as plt
from vispy import app, scene
#from ch4_util import *

# Create pattern weight matrices and subtracts diagonal mask (Inputs are in 2d arrays even if vectors)
def hebb_matrix(pattern): return pattern.T @ pattern * (1 - np.eye(len(pattern.T)))

def post_synaptic_matrix(pattern): return pattern.T @ (2 * pattern - 1) * (1 - np.eye(len(pattern.T)))

def pre_synaptic_matrix(pattern): return ( 2 * pattern.T - 1) @ pattern * (1 - np.eye(len(pattern.T)))

def Hopfield_matrix(pattern): return ( 2 * pattern.T - 1) @ (2 * pattern - 1) * (1 - np.eye(len(pattern.T))) # Covariance matrix

y_ii = np.array(
    [[1,0,1,0,1,0,1,0,1,0],
    [1,0,0,1,1,1,1,0,0,1]]
    )
y_i = np.array([y_ii[0]])

initial_pattern = [0,0,0,0,1,0,1,0,1,0]

async_order = np.random.choice(np.arange(0,len(initial_pattern), dtype=int), size=len(initial_pattern), replace=False)

rng = np.random.default_rng()

ex = np.array([[1,1,0], [0,0,1]])

'''
print(hebb_matrix(q),'\n')
print(post_synaptic_matrix(q),'\n')
print(pre_synaptic_matrix(q),'\n')
print(Hopfield_matrix(q))
'''

time_step: int = 10
y = np.zeros((time_step, len(y_ii[0]))) # Neuron state time series
y[0] = initial_pattern # Initial neural state
w = Hopfield_matrix(y_ii)
'''
# Basic synchronous network equation: $q(t)=W\cdot{y(t-1)}$
for t in range(1, time_step):
    y[t] = (w @ y[t-1] > 0).astype(int) # Remaps neural dot-product output to boolean array then converts to binary
#print(f'Sync:\n {y}\n')p
y = np.zeros((time_step, len(y_ii[0]))) # Neuron state time series
y[0] = initial_pattern # Initial neural state
'''
#print(w,'\n')
# Basic asynchronous network equation: $q_i(t)=W_{row,i}\cdot{y(t-1)}$ TODO: Async implemented, but not falling into attract pattern
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