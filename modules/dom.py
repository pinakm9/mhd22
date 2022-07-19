import tensorflow as tf 
import numpy as np

class Disk:
    def __init__(self, center=np.zeros(2), radius=1.0):
        self.center = center
        self.radius = radius
        self.n_bdry_comps = 1 

    def sample(self, n_sample):
        r = tf.sqrt(tf.random.uniform(minval=0., maxval=self.radius**2, shape=(n_sample, 1)))
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        return self.center[0] + r*tf.cos(theta), self.center[1] + r*tf.sin(theta) 

    def grid_sample(self, resolution):
        r = np.sqrt(np.linspace(start=0., stop=self.radius**2, num=resolution, endpoint=True, dtype='float32'))
        theta = np.linspace(start=0., stop=2.0*np.pi, num=resolution, endpoint=True, dtype='float32')
        r, theta = np.meshgrid(r, theta)
        r, theta = r.reshape(-1, 1), theta.reshape(-1, 1)
        return self.center[0] + r*np.cos(theta), self.center[1] + r*np.sin(theta) 


    def boundary_sample(self, n_sample):
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        c, s = tf.cos(theta), tf.sin(theta)
        x, y = self.center[0] + self.radius*c, self.center[1] + self.radius*s
        return [[x, y, c, s]]

class Annulus:
    def __init__(self, center=np.zeros(2), in_radius=0.5, out_radius=1.0):
        self.center = center
        self.in_radius = in_radius
        self.out_radius = out_radius
        self.n_bdry_comps = 2

    def sample(self, n_sample):
        r = tf.sqrt(tf.random.uniform(minval=self.in_radius, maxval=self.out_radius**2, shape=(n_sample, 1)))
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        return self.center[0] + r*tf.cos(theta), self.center[1] + r*tf.sin(theta)

    def grid_sample(self, resolution):
        r = np.sqrt(np.linspace(start=self.in_radius, stop=self.out_radius**2, num=resolution, endpoint=True, dtype='float32'))
        theta = np.linspace(start=0., stop=2.0*np.pi, num=resolution, endpoint=True, dtype='float32')
        r, theta = np.meshgrid(r, theta)
        r, theta = r.reshape(-1, 1), theta.reshape(-1, 1)
        return self.center[0] + r*np.cos(theta), self.center[1] + r*np.sin(theta)

    def boundary_sample(self, n_sample):
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        c, s = tf.cos(theta), tf.sin(theta)
        x, y = self.center[0] + self.in_radius*c, self.center[1] + self.in_radius*s
        comp1 = [x, y, -c, -s]
        x, y = self.center[0] + self.out_radius*c, self.center[1] + self.out_radius*s
        comp2 = [x, y, c, s]
        return [comp1, comp2]



class Box2D:
    def __init__(self, center, width, height):
        self.center = center 
        self.width = width 
        self.height = height
        self.n_bdry_comps = 4

    def sample(self, n_sample):
        x = tf.random.uniform(minval=-self.width/2., maxval=self.width/2., shape=(n_sample, 1))
        y = tf.random.uniform(minval=-self.width/2., maxval=self.width/2., shape=(n_sample, 1))
        return self.center[0] + x, self.center[1] + y

    def grid_sample(self, resolution):
        x = np.linspace(start=-self.width/2., stop=self.width/2., num=resolution, endpoint=True, dtype='float32')
        y = np.linspace(start=-self.width/2., stop=self.width/2., num=resolution, endpoint=True, dtype='float32')
        x, y = np.meshgrid(x, y)
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        return self.center[0] + x, self.center[1] + y

    def boundary_sample(self, n_sample):
        a, b = self.center
        w, h = self.width, self.height
        # bottom edge
        x =  tf.random.uniform(minval=a-w/2., maxval=a+w/2., shape=(n_sample, 1))
        comp1 = [x, (b-h/2.)*tf.ones_like(x), 0.*tf.ones_like(x), -1.*tf.ones_like(x)]
        # top edge
        comp3 = [x, (b+h/2.)*tf.ones_like(x), 0.*tf.ones_like(x), 1.*tf.ones_like(x)]
        # left edge
        y =  tf.random.uniform(minval=b-h/2., maxval=b+h/2., shape=(n_sample, 1))
        comp4 = [(a-w/2.)*tf.ones_like(y), y, -1.*tf.ones_like(y), 0.*tf.ones_like(x)]
        # right edge
        comp2 = [(a+w/2.)*tf.ones_like(y), y, 1.*tf.ones_like(y), 0.*tf.ones_like(x)]
        return [comp1, comp2, comp3, comp4]






