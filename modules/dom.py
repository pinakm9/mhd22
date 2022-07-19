import tensorflow as tf 
import numpy as np

class Disk:
    def __init__(self, center=np.zeros(2), radius=1.0):
        self.center = center
        self.radius = radius
        self.n_bdry_comps = 1 

    def sample(self, n_sample):
        r = tf.random.uniform(minval=0., maxval=self.radius, shape=(n_sample, 1))
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        return self.center[0] + r*tf.cos(theta), self.center[1] + r*tf.sin(theta) 

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
        r = tf.random.uniform(minval=self.in_radius, maxval=self.out_radius, shape=(n_sample, 1))
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        return self.center[0] + r*tf.cos(theta), self.center[1] + r*tf.sin(theta)

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

    def boundary_sample(self, n_smaple):
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






