import numpy as np 
import arch
import tensorflow as tf 
import time
import os
import matplotlib.pyplot as plt


class MHD_2NN:

    def __init__(self, num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu):
        self.dim = 3
        self.sys = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=10, name='system')
        self.lam = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=3, name='multiplier')
        self.domain = domain
        self.rho = rho
        self.gamma = gamma 
        self.mu0 = mu0
        self.mu = init_mu
        self.factor_mu = factor_mu


    def compute_vars(self, x, y, z):
        Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Cx = Ex + vy*Bz - vz*By
        Cy = Ey + vz*Bx - vx*Bz
        Cz = Ez + vx*By - vy*Bx 
        self.vars = [Bx, By, Bz, Cx, Cy, Cz, Ex, Ey, Ez, logp, vx, vy, vz]
        return self.vars

    def set_mu(self):
        self.mu *= self.factor_mu
        #return self.mu
        

    def loss_d(self, x, y, z):
        Bx, By, Bz, Cx, Cy, Cz, Ex, Ey, Ez, logp, vx, vy, vz = self.vars
        lamx, lamy, lamz = tf.split(self.lam(x, y, z), 3, axis=-1)
        integrand = 0.0
        integrand += 0.5 * self.rho * (vx**2 + vy**2 + vz**2)
        integrand += tf.exp(logp) / (self.gamma - 1.0) 
        integrand += (Bx**2 + By**2 + Bz**2) / (2.0*self.mu0)
        integrand += lamx*Cx + lamy*Cy + lamz*Cz
        integrand += 0.5 * self.mu * (Cx**2 + Cy**2 + Cz**2)
        return tf.reduce_mean(integrand) 

    
    def loss_E(self, x, y, z):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Ex_y = tape.gradient(Ex, y)
        Ey_x = tape.gradient(Ey, x)
        Ex_z = tape.gradient(Ex, z)
        Ez_x = tape.gradient(Ez, x)
        Ey_z = tape.gradient(Ey, z)
        Ez_y = tape.gradient(Ez, y)
        return tf.reduce_mean((Ex_y - Ey_x)**2 + (Ex_z - Ez_x)**2 + (Ey_z - Ez_y)**2)

    def loss_B(self, x, y, z):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z)
        return tf.reduce_mean(Bx_x**2 + By_y**2 + Bz_z**2)

    
    def loss_b(self, x, y, z, nx, ny, nz):
        Bx, By, Bz, Cx, Cy, Cz, Ex, Ey, Ez, logp, vx, vy, vz = self.vars
        loss = 0.0
        loss += tf.reduce_mean((nx*vx + ny*vy + nz*vz)**2)
        loss += tf.reduce_mean((nx*Bx + ny*By + nz*Bz)**2)
        loss += tf.reduce_mean(Cx**2 + Cy**2 + Cz**2)
        loss += tf.reduce_mean(Ex**2 + Ey**2 + Ez**2)
        return loss

    
    def loss_v(self, beta, domain_data, boundary_data):
        x, y, z = domain_data
        self.compute_vars(x, y, z) 
        loss = self.loss_d(x, y, z) + self.loss_B(x, y, z) + self.loss_E(x, y, z)
        for i in range(self.domain.n_bdry_comps):
            x, y, z, nx, ny, nz = boundary_data[i]
            self.compute_vars(x, y, z)
            loss += beta * self.loss_b(x, y, z, nx, ny, nz)
        return loss

    @tf.function
    def train_step_v(self, optimizer, beta, domain_data, boundary_data):
        with tf.GradientTape() as tape:
            loss = self.loss_v(beta, domain_data, boundary_data)
        grads = tape.gradient(loss, self.sys.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.sys.trainable_weights))
        return loss

    @tf.function
    def train_step_m(self, optimizer, domain_data):
        x, y, z = domain_data
        Bx, By, Bz, Cx, Cy, Cz, Ex, Ey, Ez, logp, vx, vy, vz = self.compute_vars(x, y, z)
        lamx, lamy, lamz = tf.split(self.lam(x, y, z), 3, axis=-1)
        target = tf.concat([lamx + self.mu*Cx, lamy + self.mu*Cy, lamz + self.mu*Cz], axis=-1)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.reduce_sum((self.lam(x, y, z) - target)**2, axis=-1, keepdims=True))
                
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
        ax_v = fig.add_subplot(221, projection='3d')
        ax_p = fig.add_subplot(222, projection='3d')
        ax_E = fig.add_subplot(223, projection='3d')
        ax_B = fig.add_subplot(224, projection='3d')
        x, y, z = self.domain.grid_sample(resolution)
        grid = (resolution, resolution, resolution)
        grid2 = (resolution, resolution)
        Bx, By, Bz, Cx, Cy, Cz, Ex, Ey, Ez, logp, vx, vy, vz = self.compute_vars(x, y, z)
        boundary_data = self.domain.boundary_sample(5000) 
        # for data in boundary_data:
        #     xb, yb, zb, _, _, _ = data 
        #     ax_v.scatter(xb, yb, zb, s=5)
        #     ax_E.scatter(xb, yb, zb, s=5)
        #     ax_B.scatter(xb, yb, zb, s=5)

        p = np.exp(logp.numpy().flatten())
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(p)
        ax_p.scatter(x.flatten(), y.flatten(), z.flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_p)
        ax_p.set_title('pressure')
        ax_p.grid(False)

        x, y, z = x.reshape(grid), y.reshape(grid), z.reshape(grid)
        p, q, r = vx.numpy(), vy.numpy(), vz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_v.quiver(x, y, z, p, q, r, length=0.2, colors=['blue']*len(x))
        ax_v.set_title('velocity')
        ax_v.grid(False)

        p, q, r = Ex.numpy(), Ey.numpy(), Ez.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_E.quiver(x, y, z, p, q, r, length=0.2, colors=['orange']*len(x))
        ax_E.set_title('electric field')
        ax_E.grid(False)

        p, q, r = Bx.numpy(), By.numpy(), Bz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_B.quiver(x, y, z, p, q, r, length=0.2, colors=['red']*len(x))
        ax_B.set_title('magnetic field')
        ax_B.grid(False)

        # pc = ax_p.pcolormesh(x.reshape(grid), y.reshape(grid), p.numpy().reshape(grid), cmap='inferno')
        
       
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

        

