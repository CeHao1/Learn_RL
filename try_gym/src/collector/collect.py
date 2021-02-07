# import sys
# sys.path.append('./gym')

# import gym
import numpy as np
import cvxpy as cp
import math

class cartpole_collect:
    def __init__(self, env):
        self.states = []
        self.actions = []
        self.s_dots = []
        self.x_dots = []
        self.As = []
        self.Bs = []
        self.ABs = []
        self.SAs = []

        self.gravity = env.gravity
        self.masscart = env.masscart
        self.masspole = env.masspole
        self.total_mass = env.total_mass
        self.length = env.length
        self.polemass_length = env.polemass_length
        self.force_mag = env.force_mag
        self.tau = env.tau
        self.theta_threshold_radians = env.theta_threshold_radians
        self.x_threshold = env.x_threshold
        self.Ns = 2
        self.Nc = 1
        
    def add_state(self, state):
        self.states.append(state)
        
    def add_action(self, action):
        self.actions.append(np.array([action]))
        
    def get_derivatives(self):
        self.num = len(self.actions)
        for i in range(self.num):
            self.x_dots.append(self.get_x_dot(self.states[i], self.actions[i]))
            self.s_dots.append(self.get_s_dot(self.states[i], self.states[i+1]))
            
            A,B = self.get_AB(self.states[i][[1,3]], self.s_dots[i][[1,3]], self.actions[i])
            
            self.As.append(A)
            self.Bs.append(B)
            
        self.alined_AB()
        self.alined_SA()
            
    def show(self):
        for i in range(self.num):
            # print(self.x_dots[i][[0,1]])
            # print(self.s_dots[i][[1,3]])
            # print()
            
#             print(self.states[i][[1,3]])
#             print(self.s_dots[i][[0,2]])
#             print()
            
#             print(self.As[i],'\n',self.Bs[i])
            print(self.ABs[i])
            print()
        
        
    def get_x_dot(self, state, action):
        action = action[0]
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        return np.array([xacc, thetaacc])

    def get_s_dot(self, s0, s1):
        s_dot = (s1-s0)/self.tau
        
        return s_dot      
    
    def get_AB(self, s, sd, a):
        Ns = self.Ns
        Nc = self.Nc
        
        a[0] = self.force_mag if a[0] == 1 else -self.force_mag
        
        A = cp.Variable((Ns,Ns))
        B = cp.Variable((Ns,Nc))
        
        cost = cp.Minimize(cp.sum_squares(A@s + B@a - sd))
        prob = cp.Problem(cost)
        prob.solve()
        
        return A.value, B.value
        
    def alined_AB(self):
        for i in range(self.num):
            x = np.hstack((self.As[i].flatten(),self.Bs[i].flatten()))
            self.ABs.append(x)

    def alined_SA(self):
        for i in range(self.num):
            x = np.hstack((self.states[i],self.actions[i]))
            self.SAs.append(x)

        