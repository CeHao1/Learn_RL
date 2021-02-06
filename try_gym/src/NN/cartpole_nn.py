import tensorflow as tf
import numpy as np



class cartpole_nn:
    def __init__(self):
        # self.learning_rate = 
        self.Nin = 5 # including 4 states and 1 action
        self.Nout = 6 # A=2x2, B=1x2

        self.build_model(np.array([self.Nin,]), self.Nout)
    
    def build_model(self, input_shape, output_shape):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation = 'relu', input_shape=input_shape),
            tf.keras.layers.Dense(32, activation = 'relu'),
            tf.keras.layers.Dense(output_shape)
        ])

