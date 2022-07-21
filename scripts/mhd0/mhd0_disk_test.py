# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

import mhd0
import dom
import tensorflow as tf

num_nodes = 10
num_layers = 2
domain = dom.Disk()
rho = 1.0 
gamma = 5/3.
mu0 = 1.0
init_mu = 10. 
factor_mu = 1.0006


beta = 1000.
epochs = 100 
n_sample = 1000 
save_dir = "../data/mhd0/8NN_disk"


system1 = mhd0.MHD_8NN(num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu)
learning_rate_1 = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000, 10000], [5e-3, 1e-3, 5e-4, 1e-4])
optimizers_v = [tf.keras.optimizers.Adam(learning_rate_1) for _ in range(6)]
optimizers_m = [tf.keras.optimizers.Adam(learning_rate_1) for _ in range(2)]
#system.learn(optimizers_v, optimizers_m, beta, epochs, n_sample, save_dir)
x, y = domain.sample(10)
# print(x, y)
# print('sys 1, {}'.format(system1.compute_vars(x, y)))
# print('sys 2, {}'.format(system2.compute_vars(x, y)))
# print('sys 2 loss = {}'.format(system2.loss_v(beta, domain.sample(10), domain.boundary_sample(10))))
# system2.train_step_v(optimizers_v[0], beta, domain.sample(10), domain.boundary_sample(10))
# system2.train_step_m(optimizers_v[0], domain.sample(10))
system1.learn(optimizers_v, optimizers_m, beta, epochs, n_sample, save_dir)
