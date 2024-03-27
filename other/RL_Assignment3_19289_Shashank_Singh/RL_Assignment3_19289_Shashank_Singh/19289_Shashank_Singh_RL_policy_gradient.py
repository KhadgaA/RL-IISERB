# Importing necessary library and modules

import random
import numpy as np
import scipy.stats
import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pygame
import random
import math
import matplotlib.pyplot as plt


class Warehouse_19289_Env(Env):
    def __init__(self):
        self.params = {"action_space": 4, 
                       "observation_space": (np.array([0,0]), np.array([6,5]), int),
                        "walls": np.array([[0,0], [0,1], [0,2], [0,3], [1,3], [2,3], [2,4], [2,5], [3,5], [4,5], [5,5], [5,4], [5,3], [6,3], 
                          [6,2], [6,1], [6,0], [5,0], [4,0], [3,0], [2,0], [1,0]]),
                       "player_state": np.array([1,2]),
                       "target_state": np.array([3,1]),
                       "box_state": np.array([4,3]),
                       "restricted_state": ([0,4], [0,5], [1,4], [1,5], [6,4], [6,5])
                    }
        self.action_space = Discrete(self.params["action_space"])
        self.observation_space = Box(low=self.params["observation_space"][0], high=self.params["observation_space"][1], 
                                     dtype= self.params["observation_space"][2])
        self.walls = self.params["walls"]
        
        self.state = self.params["player_state"]
        self.target_state = self.params["target_state"]
        self.box_state = self.params["box_state"]
        self.restricted_state = self.params["restricted_state"]
        
        self.action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    def step(self, action):
        info = {}
        
        done = False
        
        is_box_located = False
        
        corners_count = 0
        
        
        if self.action_map[action] == "UP":
            move_step = np.array([-1, 0])
            new_state = self.state + move_step
            
        elif self.action_map[action] == "DOWN":
            move_step = np.array([1, 0])
            new_state = self.state + move_step
            
            
        elif self.action_map[action] == "LEFT":
            move_step = np.array([0, -1])
            new_state = self.state + move_step
        
        
        elif self.action_map[action] == "RIGHT":
            move_step = np.array([0, 1])
            new_state = self.state + move_step
            
            

        if (self.walls == new_state).all(1).any():
            reward = -1
            return tuple(self.state), reward, done, info
                
        if (new_state == self.box_state).all():
            is_box_located = True
            new_box_state = new_state + move_step

            if (self.walls == new_box_state).all(1).any():
                reward = -1
                return tuple(self.state), reward, done, info
            
            if is_box_located:
                self.box_state = new_box_state
                
            for corner in np.array([[0, 1], [0, -1], [1, 1], [1, -1]]):
                if ((new_box_state + corner) == self.walls).all(1).any():
                    corners_count += 1
                    
            done = corners_count >= 2
                    
                    
        self.state = new_state                
        
        reward = int((self.box_state == self.target_state).all()) - 1
        
        done = bool((self.box_state == self.target_state).all()) or done
        
        return tuple(self.state), reward, done, info
           
                   
                                     
    def render(self):
        pass

        
    def reset(self):
        self.state = self.params["player_state"]
        self.box = self.params["box_state"]
        
        return tuple(self.state)



class LinearSoftmaxAgent(object):
    """Act with softmax policy. Features are encoded as
    phi(s, a) is a 1-hot vector of states."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.theta = np.random.random(state_size * action_size)
        self.alpha = .01
        self.gamma = .99

    def store(self, state, action, prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)

    def _phi(self, s, a):
        encoded = np.zeros([self.action_size, self.state_size])
        encoded[a] = s
        return encoded.flatten()

    def _softmax(self, s, a):
        return np.exp(self.theta.dot(self._phi(s, a)) / 100)

    def pi(self, s):
        """\pi(a | s)"""
        weights = np.empty(self.action_size)
        for a in range(self.action_size):
            weights[a] = self._softmax(s, a)
        return weights / np.sum(weights)

    def act(self, state):
        probs = self.pi(state)
        a = random.choices(range(0, self.action_size), weights=probs)
        a = a[0]
        pi = probs[a]
        return (a, pi)

    def _gradient(self, s, a):
        expected = 0
        probs = self.pi(s)
        for b in range(0, self.action_size):
            expected += probs[b] * self._phi(s, b)
        return self._phi(s, a) - expected

    def _R(self, t):
        """Reward function."""
        total = 0
        for tau in range(t, len(self.rewards)):
            total += self.gamma**(tau - t) * self.rewards[tau]
        return total

    def train(self):
        self.rewards -= np.mean(self.rewards)
        self.rewards /= np.std(self.rewards)
        for t in range(len(self.states)):
            s = self.states[t]
            a = self.actions[t]
            r = self._R(t)
            grad = self._gradient(s, a)
            self.theta = self.theta + self.alpha * r * grad
        # print(self.theta)
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []

    def getName(self):
        return 'LinearSoftmaxAgent'

    def save(self):
        pass

if __name__ == "__main__":
    SAVE_FREQUENCY = 10
    env = Warehouse_19289_Env()
    state = env.reset()
    score = 0
    episode = 0
    prev_frame = None
    state_size = 4
    action_size = env.action_space.n
    g = LinearSoftmaxEnsemble(state_size, action_size)


    MAX_EPISODES = 10000
    while episode < MAX_EPISODES:  # episode loop
       env.render()
       action, prob = g.act(state)
       state, reward, done, info = env.step(action)  # take a random action
       if done:
             reward = -10
       score += reward
       g.store(state, action, prob, reward)

       if done:
            episode += 1
            g.train()
            print('Episode: {} Score: {}'.format(episode, score))
            score = 0
            state = env.reset()
            if episode % SAVE_FREQUENCY == 0:
                 g.save()