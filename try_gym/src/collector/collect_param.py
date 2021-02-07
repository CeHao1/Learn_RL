from src.collector.collect import cartpole_collect
import numpy as np

class cartpole_collect_param(cartpole_collect):
    def __init__(self, env):
        self.params = []
        self.env = env
        super(cartpole_collect_param, self).__init__(env)


    def add_params(self, params):
        self.params.append(params)

    def get_s_dot(self, s0, s1):
        s_dot = (s1-s0)/self.tau
        # return np.array([s_dot[1],s_dot[3]])
        return s_dot[[1,3]]

    def get_derivatives(self):
        self.num = len(self.actions)
        for i in range(self.num):
            self.x_dots.append(self.get_x_dot(self.states[i], self.actions[i]))
            self.s_dots.append(self.get_s_dot(self.states[i], self.states[i+1]))

        self.alined_SA()