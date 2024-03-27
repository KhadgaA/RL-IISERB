# Importing necessary library and modules

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
    
    
    

  

    
    
if __name__ == "__main__":
    env = Warehouse_19289_Env()
    alpha = 0.85
    EPS = 0.9
    gamma = 0.95
    
    Q = {}
    xcoord = [i+1 for i in range(6)]
    ycoord = [i+1 for i in range(5)]
    
    actionSpace = [0, 1, 2, 3]
    
    stateSpace = [] 
    
    for i in xcoord:
        for j in ycoord:
            for action in actionSpace:
                Q[((i, j), action)] = 0
            stateSpace.append((i, j))
            
            
    policy = {}
    for state in stateSpace:
        policy[state] = np.random.choice(actionSpace)
        
        
    
        
    numEpisodes = 100
    for i in range(numEpisodes):
        statesActionsReturns = []
        memory = []
        print(f'...............Starting episode.....................  :::  {i}/{numEpisodes}', end='\r')
        observation = env.reset()
        done = False
        count_steps = 0
        max_steps = 10000
        while not done:
            if count_steps == max_steps:
                break
            count_steps += 1
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], action, reward))
            observation = observation_
        memory.append((observation[0], observation[1], action, reward))
        
        last = True
        for x, y, action, reward in reversed(memory):
            if last:
                last = False
            else:
                statesActionsReturns.append((x, y, action, reward))
            
            
        statesActionsReturns.reverse()
        statesActionsVisited = []
            
        
        x0, y0, action0, reward0 = statesActionsReturns[0]    
        sa = ((x0, y0), action0)
        
        for x, y, action, reward in statesActionsReturns:
            sa_ = ((x, y), action)
            if sa_ not in statesActionsVisited:
                Q[sa] += alpha * (reward + (gamma * Q[sa_]) - Q[sa])
                sa = sa_
                rand = np.random.random()
                if rand < 1 - EPS:
                    state = (x, y)
                    values = np.array([Q[(state, a)] for a in actionSpace])
                    best = np.random.choice(np.where(values == values.max())[0])
                    policy[state] = actionSpace[best]
                else:
                    policy[state] = np.random.choice(actionSpace)
                statesActionsVisited.append(sa)
                
        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0
            
    print('\n')
    print("Best selected policy :", policy)

          