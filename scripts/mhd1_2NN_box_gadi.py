# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

import mhd1
import dom
import tensorflow as tf

num_nodes = 200
num_layers = 5
domain = dom.Box3D()
rho = 1.0 
gamma = 5/3.
mu0 = 1.0
init_mu = 1. 


beta = 1000.
epochs = 500 
n_sample = 10000 
save_dir = "../data/mhd1/2NN_box"
factor_mu = (500.)**(1./epochs)

system = mhd1.MHD_2NN(num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu)
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000, 10000], [5e-3, 1e-3, 5e-4, 1e-4])
optimizer_v = tf.keras.optimizers.Adam(learning_rate)
optimizer_m = tf.keras.optimizers.Adam(learning_rate)
system.learn(optimizer_v, optimizer_m, beta, epochs, n_sample, save_dir)
system.plot(6, save_dir)