from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pygame
import random
import math


class WarehouseEnv(Env):
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Box(low=[0,0], high=[6,5],dtype= int)
        
        self.state = [1,2]
        self.target_state = [3,1]
        self.box_state = [4,3]
        
        self.action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    def step(self, action):
        info = {}
        
        done = False
        
        is_box_located = False
    
        
        
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
            return self.state, reward, done, info
                
        if (new_state == self.box_state).all():
            is_box_located = True
            new_box_state = new_state + move_step

            
            if is_box_located:
                self.box_state = new_box_state
                
                    
        self.state = new_state                
        
        reward = int((self.box_state == self.target_state).all()) - 1
        
        done = bool((self.box_state == self.target_state).all()) or done
        
        return self.state, reward, done, info
           
        
                                     
    def render(self):
        pass

        
    def reset(self):
        self.state = [1,2]
        self.box = [4,3]
        
        return self.state
    
    
    
if __name__ == "__main__":
    print("Press any button for player to move: ")
    
    env = WarehouseEnv()

    env.step(2)