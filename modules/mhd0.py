import numpy as np 
import arch
import tensorflow as tf 
import time
import os
import matplotlib.pyplot as plt

class MHD_8NN:

    def __init__(self, num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu):
        self.dim = 2
        self.vx = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='vx')
        self.vy = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='vy')
        self.Ax = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='Ax')
        self.Ay = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='Ay')
        self.phi = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='phi')
        self.p = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='p')
        self.lamx = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='lamx')
        self.lamy = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='lamy')
        self.var_nets = [self.Ax, self.Ay, self.p, self.phi, self.vx, self.vy]
        self.mul_nets = [self.lamx, self.lamy]
        self.domain = domain
        self.rho = rho
        self.gamma = gamma 
        self.mu0 = mu0
        self.mu = init_mu
        self.factor_mu = factor_mu


    @tf.function
    def B(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            Ax = self.Ax(x, y)
            Ay = self.Ay(x, y)
        Ax_y = tape.gradient(Ax, y)
        Ay_x = tape.gradient(Ay, x)
        return Ay_x - Ax_y 

    @tf.function
    def E(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            phi = self.phi(x, y)
        Ex = - tape.gradient(phi, x)
        Ey = - tape.gradient(phi, y)
        return Ex, Ey, phi


    def compute_vars(self, x, y):
        vx, vy = self.vx(x, y), self.vy(x, y) 
        Ex, Ey, phi = self.E(x, y)
        Bz = self.B(x, y)
        p = self.p(x, y)
        Cx, Cy = Ex + vy*Bz, Ey - vx*Bz
        self.vars = [Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy]
        return self.vars

    def set_mu(self):
        self.mu *= self.factor_mu
        #return self.mu
        

    def loss_d(self, x, y):
        Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy = self.vars
        lamx, lamy = self.lamx(x, y), self.lamy(x, y)
        integrand = 0.0
        integrand += 0.5 * self.rho * (vx**2 + vy**2)
        integrand += p / (self.gamma - 1.0) 
        integrand += Bz**2 / (2.0*self.mu0)
        integrand -= lamx*Cx + lamy*Cy
        integrand += 0.5 * self.mu * (Cx**2 + Cy**2)
        return tf.reduce_mean(integrand) 

    
    
    def loss_b(self, x, y, nx, ny):
        Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy = self.vars
        loss = 0.0
        loss += tf.reduce_mean((nx*vx + ny*vy)**2)
        loss += tf.reduce_mean(Cx**2 + Cy**2)
        loss += tf.reduce_mean(Ex**2 + Ey**2)
        return loss

    
    def loss_v(self, beta, domain_data, boundary_data):
        x, y = domain_data
        self.compute_vars(x, y)
        loss = self.loss_d(x, y)
        for i in range(self.domain.n_bdry_comps):
            x, y, nx, ny = boundary_data[i]
            self.compute_vars(x, y)
            loss += beta * self.loss_b(x, y, nx, ny)
        return loss

    
    @tf.function
    def train_step_v(self, optimizers_v, beta, domain_data, boundary_data):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_v(beta, domain_data, boundary_data)
        for i, net in enumerate(self.var_nets):
            grads = tape.gradient(loss, net.trainable_weights)
            optimizers_v[i].apply_gradients(zip(grads, net.trainable_weights))
        return loss

    @tf.function
    def train_step_m(self, optimizers_m, domain_data):
        x, y = domain_data
        Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy = self.compute_vars(x, y)
        lamx, lamy = self.lamx(x, y), self.lamy(x, y)
        targetx, targety = lamx - self.mu*Cx, lamy - self.mu*Cy
        with tf.GradientTape(persistent=True) as tape:
            loss = tf.reduce_sum((self.lamx(x, y) - targetx)**2 + (self.lamy(x, y) - targety)**2)
                
        for i, net in enumerate(self.mul_nets):
            grads = tape.gradient(loss, net.trainable_weights)
            optimizers_m[i].apply_gradients(zip(grads, net.trainable_weights))
        
        self.set_mu()
        return loss

    
    def learn(self, optimizers_v, optimizers_m, beta, epochs, n_sample, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        domain_data = self.domain.sample(n_sample)
        boundary_data = self.domain.boundary_sample(n_sample)
        print("{:>6}{:>12}{:>12}{:>18}".format('epoch', 'loss_v', 'loss_m', 'runtime(s)'))
        start = time.time()
        for epoch in range(epochs):
            self.compute_vars(*domain_data)
            l1 = self.train_step_v(optimizers_v, beta, domain_data, boundary_data)
            l2 = self.train_step_m(optimizers_m, domain_data)
            if epoch % 10 == 0:
                print('{:6d}{:12.6f}{:16.6f}{:12.4f}'.format(epoch, l1, l2, time.time()-start))
                domain_data = self.domain.sample(n_sample)
                boundary_data = self.domain.boundary_sample(n_sample)
        

        for net in self.var_nets:
            net.save_weights('{}/{}'.format(save_dir, net.name))

        for net in self.mul_nets:
            net.save_weights('{}/{}'.format(save_dir, net.name))

        






class MHD_2NN:

    def __init__(self, num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu):
        self.dim = 2
        self.sys = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=6, name='system')
        self.lam = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=2, name='multiplier')
        self.domain = domain
        self.rho = rho
        self.gamma = gamma 
        self.mu0 = mu0
        self.mu = init_mu
        self.factor_mu = factor_mu


    @tf.function
    def B(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            Ax, Ay, p, phi, vx, vy = tf.split(self.sys(x, y), 6, axis=-1)
        Ax_y = tape.gradient(Ax, y)
        Ay_x = tape.gradient(Ay, x)
        return Ay_x - Ax_y 

    @tf.function
    def E(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            Ax, Ay, p, phi, vx, vy = tf.split(self.sys(x, y), 6, axis=-1)
        Ex = - tape.gradient(phi, x)
        Ey = - tape.gradient(phi, y)
        return Ex, Ey


    def compute_vars(self, x, y):
        Ax, Ay, p, phi, vx, vy = tf.split(self.sys(x, y), 6, axis=-1)
        Ex, Ey = self.E(x, y)
        Bz = self.B(x, y)
        Cx, Cy = Ex + vy*Bz, Ey - vx*Bz
        self.vars = [Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy]
        return self.vars

    def set_mu(self):
        self.mu *= self.factor_mu
        #return self.mu
        

    def loss_d(self, x, y):
        Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy = self.vars
        lamx, lamy = tf.split(self.lam(x, y), 2, axis=-1)
        integrand = 0.0
        integrand += 0.5 * self.rho * (vx**2 + vy**2)
        integrand += p / (self.gamma - 1.0) 
        integrand += Bz**2 / (2.0*self.mu0)
        integrand += lamx*Cx + lamy*Cy
        integrand += 0.5 * self.mu * (Cx**2 + Cy**2)
        return tf.reduce_mean(integrand) 

    
    
    def loss_b(self, x, y, nx, ny):
        Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy = self.vars
        loss = 0.0
        loss += tf.reduce_mean((nx*vx + ny*vy)**2)
        loss += tf.reduce_mean(Cx**2 + Cy**2)
        loss += tf.reduce_mean(Ex**2 + Ey**2)
        return loss

    
    def loss_v(self, beta, domain_data, boundary_data):
        x, y = domain_data
        self.compute_vars(x, y)
        loss = self.loss_d(x, y)
        for i in range(self.domain.n_bdry_comps):
            x, y, nx, ny = boundary_data[i]
            self.compute_vars(x, y)
            loss += beta * self.loss_b(x, y, nx, ny)
        return loss

    # @tf.function
    def train_step_v(self, optimizer, beta, domain_data, boundary_data):
        with tf.GradientTape() as tape:
            loss = self.loss_v(beta, domain_data, boundary_data)
        grads = tape.gradient(loss, self.sys.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.sys.trainable_weights))
        return loss

    @tf.function
    def train_step_m(self, optimizer, domain_data):
        x, y = domain_data
        Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy = self.compute_vars(x, y)
        lamx, lamy = tf.split(self.lam(x, y), 2, axis=-1)
        target = tf.concat([lamx + self.mu*Cx, lamy + self.mu*Cy], axis=-1)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.reduce_sum((self.lam(x, y) - target)**2, axis=-1, keepdims=True))
                
        grads = tape.gradient(loss, self.lam.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.lam.trainable_weights))
        
        self.set_mu()
        return loss

    
    def learn(self, optimizer_v, optimizer_m, beta, epochs, n_sample, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        domain_data = self.domain.sample(n_sample)
        boundary_data = self.domain.boundary_sample(n_sample)
        heading = "{:>6}{:>12}{:>12}{:>18}".format('epoch', 'loss_v', 'loss_m', 'runtime(s)')
        print(heading)
        start = time.time()
        with open('{}/training_log.txt'.format(save_dir), 'w') as log:
            log.write(heading + '\n')
            for epoch in range(epochs):
                self.compute_vars(*domain_data)
                l1 = self.train_step_v(optimizer_v, beta, domain_data, boundary_data)
                l2 = self.train_step_m(optimizer_m, domain_data)
                if epoch % 10 == 0:
                    stdout = '{:6d}{:12.6f}{:16.6f}{:12.4f}'.format(epoch, l1, l2, time.time()-start)
                    print(stdout)
                    log.write(stdout + '\n')
                    domain_data = self.domain.sample(n_sample)
                    boundary_data = self.domain.boundary_sample(n_sample)

        self.sys.save_weights('{}/{}'.format(save_dir, self.sys.name))
        self.lam.save_weights('{}/{}'.format(save_dir, self.lam.name))

    
    def plot(self, resolution, save_dir):
        self.sys.load_weights('{}/{}'.format(save_dir, self.sys.name)).expect_partial()
        self.lam.load_weights('{}/{}'.format(save_dir, self.lam.name)).expect_partial()
        fig = plt.figure(figsize=(16, 16))
        ax_v = fig.add_subplot(221)
        ax_p = fig.add_subplot(222)
        ax_E = fig.add_subplot(223)
        ax_B = fig.add_subplot(224)
        x, y = self.domain.grid_sample(resolution)
        grid = (resolution, resolution)
        Bz, Ex, Ey, Cx, Cy, p, phi, vx, vy = self.compute_vars(x, y)
        boundary_data = self.domain.boundary_sample(5000) 
        for data in boundary_data:
            xb, yb, _, _ = data 
            ax_v.scatter(xb, yb, s=5)
            ax_E.scatter(xb, yb, s=5)
        ax_v.quiver(x.reshape(grid), y.reshape(grid), vx.numpy().reshape(grid), vy.numpy().reshape(grid))
        ax_v.set_title('velocity')
        # ax_v.streamplot(x.flatten(), y.flatten(), vx.numpy().flatten(), vy.numpy().flatten())
        # ax_v.set_title('velocity')
        ax_E.quiver(x.reshape(grid), y.reshape(grid), Ex.numpy().reshape(grid), Ey.numpy().reshape(grid))
        ax_E.set_title('electric field')
        # ax_E.streamplot(x.flatten(), y.flatten(), Ex.numpy().flatten(), Ey.numpy().flatten())
        # ax_E.set_title('electric field')
        pc = ax_p.pcolormesh(x.reshape(grid), y.reshape(grid), p.numpy().reshape(grid), cmap='inferno')
        fig.colorbar(pc, ax=ax_p)
        ax_p.set_title('pressure')
        ax_p.set_aspect('equal')
        pc = ax_B.pcolormesh(x.reshape(grid), y.reshape(grid), Bz.numpy().reshape(grid), cmap='inferno')
        fig.colorbar(pc, ax=ax_B)
        ax_B.set_title('magnetic field')
        ax_B.set_aspect('equal')
        plt.savefig('{}/solution.png'.format(save_dir))



class MHD_2NN_0:

    def __init__(self, num_nodes, num_layers, domain, rho, gamma, mu0):
        self.dim = 2
        self.sys = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=6, name='system')
        self.domain = domain
        self.rho = rho
        self.gamma = gamma
        self.mu0 = mu0


    @tf.function
    def B(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            Ax, Ay, p, phi, vx, vy = tf.split(self.sys(x, y), 6, axis=-1)
        Ax_y = tape.gradient(Ax, y)
        Ay_x = tape.gradient(Ay, x)
        return Ay_x - Ax_y 

    @tf.function
    def E(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            Ax, Ay, p, phi, vx, vy = tf.split(self.sys(x, y), 6, axis=-1)
        Ex = - tape.gradient(phi, x)
        Ey = - tape.gradient(phi, y)
        return Ex, Ey


    def compute_vars(self, x, y):
        Ax, Ay, logp, phi, vx, vy = tf.split(self.sys(x, y), 6, axis=-1)
        Ex, Ey = self.E(x, y)
        Bz = self.B(x, y)
        Cx, Cy = Ex + vy*Bz, Ey - vx*Bz
        self.vars = [Bz, Ex, Ey, Cx, Cy, logp, phi, vx, vy]
        return self.vars
        

    def loss_d(self, x, y):
        Bz, Ex, Ey, Cx, Cy, logp, phi, vx, vy = self.vars
        integrand = 0.0
        integrand += 0.5 * self.rho * (vx**2 + vy**2)
        integrand += tf.exp(logp) / (self.gamma - 1.0) 
        integrand += Bz**2 / (2.0*self.mu0)
        integrand += (Cx**2 + Cy**2)
        return tf.reduce_mean(integrand) 

    
    
    def loss_b(self, x, y, nx, ny):
        Bz, Ex, Ey, Cx, Cy, logp, phi, vx, vy = self.vars
        loss = 0.0
        loss += tf.reduce_mean((nx*vx + ny*vy)**2)
        loss += tf.reduce_mean(Cx**2 + Cy**2)
        loss += tf.reduce_mean(Ex**2 + Ey**2)
        return loss

    
    def loss_v(self, beta, domain_data, boundary_data):
        x, y = domain_data
        self.compute_vars(x, y)
        loss = self.loss_d(x, y)
        for i in range(self.domain.n_bdry_comps):
            x, y, nx, ny = boundary_data[i]
            self.compute_vars(x, y)
            loss += beta * self.loss_b(x, y, nx, ny)
        return loss

    # @tf.function
    def train_step_v(self, optimizer, beta, domain_data, boundary_data):
        with tf.GradientTape() as tape:
            loss = self.loss_v(beta, domain_data, boundary_data)
        grads = tape.gradient(loss, self.sys.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.sys.trainable_weights))
        return loss

    
    def learn(self, optimizer_v, beta, epochs, n_sample, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        domain_data = self.domain.sample(n_sample)
        boundary_data = self.domain.boundary_sample(n_sample)
        heading = "{:>6}{:>12}{:>18}".format('epoch', 'loss_v', 'runtime(s)')
        print(heading)
        start = time.time()
        with open('{}/training_log.txt'.format(save_dir), 'w') as log:
            log.write(heading + '\n')
            for epoch in range(epochs):
                self.compute_vars(*domain_data)
                l1 = self.train_step_v(optimizer_v, beta, domain_data, boundary_data)
                if epoch % 10 == 0:
                    stdout = '{:6d}{:12.6f}{:12.4f}'.format(epoch, l1, time.time()-start)
                    print(stdout)
                    log.write(stdout + '\n')
                    domain_data = self.domain.sample(n_sample)
                    boundary_data = self.domain.boundary_sample(n_sample)

        self.sys.save_weights('{}/{}'.format(save_dir, self.sys.name))
        
 

    
    def plot(self, resolution, save_dir):
        self.sys.load_weights('{}/{}'.format(save_dir, self.sys.name)).expect_partial()
        fig = plt.figure(figsize=(16, 16))
        ax_v = fig.add_subplot(221)
        ax_p = fig.add_subplot(222)
        ax_E = fig.add_subplot(223)
        ax_B = fig.add_subplot(224)
        x, y = self.domain.grid_sample(resolution)
        grid = (resolution, resolution)
        Bz, Ex, Ey, Cx, Cy, logp, phi, vx, vy = self.compute_vars(x, y)
        boundary_data = self.domain.boundary_sample(5000) 
        for data in boundary_data:
            xb, yb, _, _ = data 
            ax_v.scatter(xb, yb, s=5)
            ax_E.scatter(xb, yb, s=5)
        ax_v.quiver(x.reshape(grid), y.reshape(grid), vx.numpy().reshape(grid), vy.numpy().reshape(grid))
        ax_v.set_title('velocity')
        # ax_v.streamplot(x.flatten(), y.flatten(), vx.numpy().flatten(), vy.numpy().flatten())
        # ax_v.set_title('velocity')
        ax_E.quiver(x.reshape(grid), y.reshape(grid), Ex.numpy().reshape(grid), Ey.numpy().reshape(grid))
        ax_E.set_title('electric field')
        # ax_E.streamplot(x.flatten(), y.flatten(), Ex.numpy().flatten(), Ey.numpy().flatten())
        # ax_E.set_title('electric field')
        pc = ax_p.pcolormesh(x.reshape(grid), y.reshape(grid), np.exp(logp.numpy()).reshape(grid), cmap='inferno')
        fig.colorbar(pc, ax=ax_p)
        ax_p.set_title('pressure')
        ax_p.set_aspect('equal')
        pc = ax_B.pcolormesh(x.reshape(grid), y.reshape(grid), Bz.numpy().reshape(grid), cmap='inferno')
        fig.colorbar(pc, ax=ax_B)
        ax_B.set_title('magnetic field')
        ax_B.set_aspect('equal')
        plt.savefig('{}/solution.png'.format(save_dir))

        

