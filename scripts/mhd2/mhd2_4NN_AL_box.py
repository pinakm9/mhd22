# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

import mhd2 as mhd
import dom
import tensorflow as tf

num_nodes = 100
num_layers = 3
domain = dom.Box3D()
rho = 1.0 
gamma = 5/3.
mu0 = 1.0
init_mu = 1. 


beta = 1000.
epochs = 10000
n_sample = 1000 
save_dir = "../../data/mhd2/4NN_AL_box"
factor_mu = (500.)**(1./epochs)

system = mhd.MHD_4NN_AL(num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu)
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000, 10000], [5e-3, 1e-3, 5e-4, 1e-4])
optimizer_v = tf.keras.optimizers.Adam(learning_rate)
optimizers_m = [tf.keras.optimizers.Adam(learning_rate) for _ in range(3)]
system.learn(optimizer_v, optimizers_m, beta, epochs, n_sample, save_dir)
system.plot(6, save_dir)
system.plot_constraints(6, save_dir)