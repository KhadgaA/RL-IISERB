import matplotlib.pyplot as plt
import numpy as np
import bandit_envi as be

arms = be.bandit_env([2.5, -3.5, 1.0, 5.0, -2.5], [0.33, 1.0, 0.66, 1.98, 1.65])

def ep_greedy(arms,ep = 0.1, N = 1000,return_data = 0,show_optimal = True):
    
    def action(a):
        reward = arms.pull(a)
        return reward

    def update(reward,a,timestep):
        mean[a] = mean[a] + 1/timestep[a]*(reward - mean[a])
        return mean[a]
    
    bandits = arms.n
    mean = np.zeros(bandits)
    returns = []
    timestep =np.zeros(bandits)
    actions = np.arange(bandits)
    act3r = []
    for i in range(1,N+1):
        if np.random.rand()<=ep:
            a = np.random.choice(actions)
        else: 
            a = np.argmax(mean)
        timestep[a]+=1
        reward = action(a)
        update(reward,a,timestep)
        returns.append(reward)
        if show_optimal==True:
            act3 = action(3)
            act3r.append(act3)
    
            
    cumulative_average = np.cumsum(returns) / (np.arange(N) + 1)
    plt.plot(cumulative_average)
    
    if show_optimal==True:
        cumulative_average3 = np.cumsum(act3r) / (np.arange(N) + 1)
        plt.plot(cumulative_average3)
    
    if return_data == True:
        return returns
ep_greedy(arms)