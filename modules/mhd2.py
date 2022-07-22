import numpy as np 
import arch
import tensorflow as tf 
import time
import os
import matplotlib.pyplot as plt


class MHD_4NN_AL:

    def __init__(self, num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu):
        self.dim = 3
        self.sys = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=10, name='system')
        self.lamC = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=3, name='multiplier_C')
        self.lamB = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=1, name='multiplier_B')
        self.lamE = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=3, name='multiplier_E')
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
        lamx, lamy, lamz = tf.split(self.lamC(x, y, z), 3, axis=-1)
        integrand = 0.0
        integrand += 0.5 * self.rho * (vx**2 + vy**2 + vz**2)
        integrand += tf.exp(logp) / (self.gamma - 1.0) 
        integrand += (Bx**2 + By**2 + Bz**2) / (2.0*self.mu0)
        integrand -= lamx*Cx + lamy*Cy + lamz*Cz
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
        lamx, lamy, lamz = tf.split(self.lamE(x, y, z), 3, axis=-1)
        penalty =  0.5 * self.mu * ((Ex_y - Ey_x)**2 + (Ex_z - Ez_x)**2 + (Ey_z - Ez_y)**2)
        lm_term = lamx*(Ex_y - Ey_x) + lamy*(Ex_z - Ez_x) + lamz*(Ey_z - Ez_y)
        return tf.reduce_mean(penalty - lm_term)

    
    def loss_B(self, x, y, z):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z)
        lam = self.lamB(x, y, z)
        penalty = 0.5 * self.mu * (Bx_x + By_y + Bz_z)**2
        lm_term = lam * (Bx_x + By_y + Bz_z)
        return tf.reduce_mean(penalty - lm_term)

    
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
    def train_step_lamC(self, optimizer, domain_data):
        x, y, z = domain_data
        Bx, By, Bz, Cx, Cy, Cz, Ex, Ey, Ez, logp, vx, vy, vz = self.compute_vars(x, y, z)
        lamx, lamy, lamz = tf.split(self.lamC(x, y, z), 3, axis=-1)
        target = tf.concat([lamx - self.mu*Cx, lamy - self.mu*Cy, lamz - self.mu*Cz], axis=-1)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.reduce_sum((self.lamC(x, y, z) - target)**2, axis=-1, keepdims=True))
                
        grads = tape.gradient(loss, self.lamC.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.lamC.trainable_weights))
        return loss

    @tf.function
    def train_step_lamE(self, optimizer, domain_data):
        x, y, z = domain_data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Ex_y = tape.gradient(Ex, y)
        Ey_x = tape.gradient(Ey, x)
        Ex_z = tape.gradient(Ex, z)
        Ez_x = tape.gradient(Ez, x)
        Ey_z = tape.gradient(Ey, z)
        Ez_y = tape.gradient(Ez, y)
        lamx, lamy, lamz = tf.split(self.lamE(x, y, z), 3, axis=-1)
        target = tf.concat([lamx - self.mu*(Ex_y - Ey_x), lamy - self.mu*(Ex_z - Ez_x), lamz - self.mu*(Ey_z - Ez_y)], axis=-1)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.reduce_sum((self.lamE(x, y, z) - target)**2, axis=-1, keepdims=True))
                
        grads = tape.gradient(loss, self.lamE.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.lamE.trainable_weights))
        return loss


    @tf.function
    def train_step_lamB(self, optimizer, domain_data):
        x, y, z = domain_data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z)
        lam = self.lamB(x, y, z)
        target = lam - self.mu*(Bx_x + By_y + Bz_z)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((self.lamB(x, y, z) - target)**2)
                
        grads = tape.gradient(loss, self.lamB.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.lamB.trainable_weights))
        return loss

    
    def learn(self, optimizer_v, optimizers_m, beta, epochs, n_sample, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        domain_data = self.domain.sample(n_sample)
        boundary_data = self.domain.boundary_sample(n_sample)
        heading = "{:>6}{:>12}{:>12}{:>12}{:>12}{:>18}"\
                  .format('epoch', 'loss_v', 'loss_lamB', 'loss_lamC', 'loss_lamE', 'runtime(s)')
        print(heading)
        start = time.time()
        with open('{}/training_log.txt'.format(save_dir), 'w') as log:
            log.write(heading + '\n')
            for epoch in range(epochs):
                self.compute_vars(*domain_data)
                l1 = self.train_step_v(optimizer_v, beta, domain_data, boundary_data)
                l2 = self.train_step_lamB(optimizers_m[0], domain_data)
                l3 = self.train_step_lamC(optimizers_m[1], domain_data)
                l4 = self.train_step_lamE(optimizers_m[2], domain_data)
                self.set_mu()
                if epoch % 10 == 0:
                    stdout = '{:6d}{:12.6f}{:12.6f}{:12.6f}{:16.6f}{:12.4f}'\
                             .format(epoch, l1, l2, l3, l4, time.time()-start)
                    print(stdout)
                    log.write(stdout + '\n')
                    domain_data = self.domain.sample(n_sample)
                    boundary_data = self.domain.boundary_sample(n_sample)

        self.sys.save_weights('{}/{}'.format(save_dir, self.sys.name))
        self.lamB.save_weights('{}/{}'.format(save_dir, self.lamB.name))
        self.lamE.save_weights('{}/{}'.format(save_dir, self.lamE.name))
        self.lamC.save_weights('{}/{}'.format(save_dir, self.lamC.name))

    
    def plot(self, resolution, save_dir):
        self.sys.load_weights('{}/{}'.format(save_dir, self.sys.name)).expect_partial()
        self.lamC.load_weights('{}/{}'.format(save_dir, self.lamC.name)).expect_partial()
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
        ax_p.set_title('pressure', fontsize=20)
        ax_p.grid(False)

        x, y, z = x.reshape(grid), y.reshape(grid), z.reshape(grid)
        p, q, r = vx.numpy(), vy.numpy(), vz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_v.quiver(x, y, z, p, q, r, length=0.2, colors=['blue']*len(x))
        ax_v.set_title('velocity', fontsize=20)
        ax_v.grid(False)

        p, q, r = Ex.numpy(), Ey.numpy(), Ez.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_E.quiver(x, y, z, p, q, r, length=0.2, colors=['orange']*len(x))
        ax_E.set_title('electric field', fontsize=20)
        ax_E.grid(False)

        p, q, r = Bx.numpy(), By.numpy(), Bz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_B.quiver(x, y, z, p, q, r, length=0.2, colors=['red']*len(x))
        ax_B.set_title('magnetic field', fontsize=20)
        ax_B.grid(False)
        plt.savefig('{}/solution.png'.format(save_dir))

    def plot_constraints(self, resolution, save_dir):
        self.sys.load_weights('{}/{}'.format(save_dir, self.sys.name)).expect_partial()
        self.lamB.load_weights('{}/{}'.format(save_dir, self.lamB.name)).expect_partial()
        self.lamE.load_weights('{}/{}'.format(save_dir, self.lamE.name)).expect_partial()
        self.lamC.load_weights('{}/{}'.format(save_dir, self.lamC.name)).expect_partial()
        
        fig = plt.figure(figsize=(16, 16))
        ax_C = fig.add_subplot(221, projection='3d')
        ax_p = fig.add_subplot(222, projection='3d')
        ax_E = fig.add_subplot(223, projection='3d')
        ax_B = fig.add_subplot(224, projection='3d')
        x, y, z = self.domain.grid_sample(resolution)
        x = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
        y = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)
        z = tf.convert_to_tensor(z.reshape(-1, 1), dtype=tf.float32)
        grid = (resolution, resolution, resolution)
        grid2 = (resolution, resolution)

        # divergence of B
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z)
        divB = (Bx_x + By_y + Bz_z).numpy().flatten()
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(divB)
        ax_B.scatter(x.numpy().flatten(), y.numpy().flatten(), z.numpy().flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_B)
        ax_B.set_title('$\\nabla\\cdot B$', fontsize=20)
        ax_B.grid(False)

        # curl of E
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Ex_y = tape.gradient(Ex, y)
        Ey_x = tape.gradient(Ey, x)
        Ex_z = tape.gradient(Ex, z)
        Ez_x = tape.gradient(Ez, x)
        Ey_z = tape.gradient(Ey, z)
        Ez_y = tape.gradient(Ez, y)

        curlE = np.sqrt(((Ex_y - Ey_x)**2 + (Ex_z - Ez_x)**2 + (Ey_z - Ez_y)**2).numpy().flatten())
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(curlE)
        ax_E.scatter(x.numpy().flatten(), y.numpy().flatten(), z.numpy().flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_E)
        ax_E.set_title('$|\\nabla\\times E|$', fontsize=20)
        ax_E.grid(False)


        # Ohm's law
        Bx, By, Bz, Cx, Cy, Cz, Ex, Ey, Ez, logp, vx, vy, vz = self.compute_vars(x, y, z)

        C = np.sqrt((Cx**2 + Cy**2 + Cz**2).numpy().flatten())
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(C)
        ax_C.scatter(x.numpy().flatten(), y.numpy().flatten(), z.numpy().flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_C)
        ax_C.set_title('$|E+v\\times B|$', fontsize=20)
        ax_C.grid(False)
    
        plt.savefig('{}/constraints.png'.format(save_dir))





















class MHD_1NN:

    def __init__(self, num_nodes, num_layers, domain, rho, gamma, mu0, init_mu, factor_mu):
        self.dim = 3
        self.sys = arch.LSTMForgetNet(num_nodes=num_nodes, num_layers=num_layers, out_dim=10, name='system')
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
        integrand = 0.0
        integrand += 0.5 * self.rho * (vx**2 + vy**2 + vz**2)
        integrand += tf.exp(logp) / (self.gamma - 1.0) 
        integrand += (Bx**2 + By**2 + Bz**2) / (2.0*self.mu0)
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
        penalty =  0.5 * self.mu * ((Ex_y - Ey_x)**2 + (Ex_z - Ez_x)**2 + (Ey_z - Ez_y)**2)
        return tf.reduce_mean(penalty)

    
    def loss_B(self, x, y, z):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z)
        penalty = 0.5 * self.mu * (Bx_x + By_y + Bz_z)**2
        return tf.reduce_mean(penalty)

    
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

    
    def learn(self, optimizer_v, beta, epochs, n_sample, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        domain_data = self.domain.sample(n_sample)
        boundary_data = self.domain.boundary_sample(n_sample)
        heading = "{:>6}{:>12}{:>18}"\
                  .format('epoch', 'loss_v', 'runtime(s)')
        print(heading)
        start = time.time()
        with open('{}/training_log.txt'.format(save_dir), 'w') as log:
            log.write(heading + '\n')
            for epoch in range(epochs):
                self.compute_vars(*domain_data)
                l1 = self.train_step_v(optimizer_v, beta, domain_data, boundary_data)
                self.set_mu()
                if epoch % 10 == 0:
                    stdout = '{:6d}{:16.6f}{:12.4f}'\
                             .format(epoch, l1, time.time()-start)
                    print(stdout)
                    log.write(stdout + '\n')
                    domain_data = self.domain.sample(n_sample)
                    boundary_data = self.domain.boundary_sample(n_sample)

        self.sys.save_weights('{}/{}'.format(save_dir, self.sys.name))

    
    def plot(self, resolution, save_dir):
        self.sys.load_weights('{}/{}'.format(save_dir, self.sys.name)).expect_partial()
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
        ax_p.set_title('pressure', fontsize=20)
        ax_p.grid(False)

        x, y, z = x.reshape(grid), y.reshape(grid), z.reshape(grid)
        p, q, r = vx.numpy(), vy.numpy(), vz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_v.quiver(x, y, z, p, q, r, length=0.2, colors=['blue']*len(x))
        ax_v.set_title('velocity', fontsize=20)
        ax_v.grid(False)

        p, q, r = Ex.numpy(), Ey.numpy(), Ez.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_E.quiver(x, y, z, p, q, r, length=0.2, colors=['orange']*len(x))
        ax_E.set_title('electric field', fontsize=20)
        ax_E.grid(False)

        p, q, r = Bx.numpy(), By.numpy(), Bz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(grid), q.reshape(grid), r.reshape(grid)
        ax_B.quiver(x, y, z, p, q, r, length=0.2, colors=['red']*len(x))
        ax_B.set_title('magnetic field', fontsize=20)
        ax_B.grid(False)
        plt.savefig('{}/solution.png'.format(save_dir))

    def plot_constraints(self, resolution, save_dir):
        self.sys.load_weights('{}/{}'.format(save_dir, self.sys.name)).expect_partial()
        
        fig = plt.figure(figsize=(16, 16))
        ax_C = fig.add_subplot(221, projection='3d')
        ax_p = fig.add_subplot(222, projection='3d')
        ax_E = fig.add_subplot(223, projection='3d')
        ax_B = fig.add_subplot(224, projection='3d')
        x, y, z = self.domain.grid_sample(resolution)
        x = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
        y = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)
        z = tf.convert_to_tensor(z.reshape(-1, 1), dtype=tf.float32)
        grid = (resolution, resolution, resolution)
        grid2 = (resolution, resolution)

        # divergence of B
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z)
        divB = (Bx_x + By_y + Bz_z).numpy().flatten()
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(divB)
        ax_B.scatter(x.numpy().flatten(), y.numpy().flatten(), z.numpy().flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_B)
        ax_B.set_title('$\\nabla\\cdot B$', fontsize=20)
        ax_B.grid(False)

        # curl of E
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz, Ex, Ey, Ez, logp, vx, vy, vz = tf.split(self.sys(x, y, z), 10, axis=-1)
        Ex_y = tape.gradient(Ex, y)
        Ey_x = tape.gradient(Ey, x)
        Ex_z = tape.gradient(Ex, z)
        Ez_x = tape.gradient(Ez, x)
        Ey_z = tape.gradient(Ey, z)
        Ez_y = tape.gradient(Ez, y)

        curlE = np.sqrt(((Ex_y - Ey_x)**2 + (Ex_z - Ez_x)**2 + (Ey_z - Ez_y)**2).numpy().flatten())
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(curlE)
        ax_E.scatter(x.numpy().flatten(), y.numpy().flatten(), z.numpy().flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_E)
        ax_E.set_title('$|\\nabla\\times E|$', fontsize=20)
        ax_E.grid(False)


        # Ohm's law
        Bx, By, Bz, Cx, Cy, Cz, Ex, Ey, Ez, logp, vx, vy, vz = self.compute_vars(x, y, z)

        C = np.sqrt((Cx**2 + Cy**2 + Cz**2).numpy().flatten())
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(C)
        ax_C.scatter(x.numpy().flatten(), y.numpy().flatten(), z.numpy().flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_C)
        ax_C.set_title('$|E+v\\times B|$', fontsize=20)
        ax_C.grid(False)
    
        plt.savefig('{}/constraints.png'.format(save_dir))



        

