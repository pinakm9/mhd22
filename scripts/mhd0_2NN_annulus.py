# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
from cv2 import threshold
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

import mhd0
import dom
import tensorflow as tf

num_nodes = 100
num_layers = 3
domain = dom.Annulus()
rho = 1.0 
gamma = 5/3.
mu0 = 1.0
init_mu = 1. 


beta = 1000.
epochs = 100 
n_sample = 1000 
save_dir = "../data/mhd0/2NN_annulus"
factor_mu = (10.)**(1./epochs)

system2 = mhd0.MHD_2NN(num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu)
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000, 10000], [5e-3, 1e-3, 5e-4, 1e-4])
optimizer_v = tf.keras.optimizers.Adam(learning_rate)
optimizer_m = tf.keras.optimizers.Adam(learning_rate)
system2.learn(optimizer_v, optimizer_m, beta, epochs, n_sample, save_dir)
system2.plot(30, save_dir)