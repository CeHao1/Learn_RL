import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class cartpole_nn:
    def __init__(self, train_data, train_goal):
        self.train_data = train_data
        self.train_goal = train_goal

        self.build_model(train_data[0].shape, train_goal.shape[1])
    
    def build_model(self, input_shape, output_shape):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation = 'relu', input_shape=input_shape),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dense(output_shape)
        ])

        # self.model = tf.keras.models.Sequential([
        #     tf.keras.layers.Dense(64, activation = 'relu', input_shape=input_shape),
        #     tf.keras.layers.Dense(32, activation = 'relu'),
        #     tf.keras.layers.Dense(output_shape)
        # ])

        self.model.compile(
            optimizer = 'Adam',
            loss = 'mse',
            metrics = ['mse']
        )

    def train_model(self, epochs = 100, batch_size=64, verbose = 2):
        self.history = self.model.fit(self.train_data, self.train_goal,
        epochs = epochs, batch_size = batch_size,
        verbose = verbose, validation_split = 0.15)

    def plot(self):
        history = self.history

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss vs. epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.show()

    def eval(self, test_data, test_goal):
        loss, _ = self.model.evaluate(test_data, test_goal)
        print('Error: ', 100.0 * np.sqrt(loss)/np.average(abs(test_goal)),' %')