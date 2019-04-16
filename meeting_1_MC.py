import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pandas import ewma


# In[]
env = gym.make("CliffWalking-v0")

n_actions = env.action_space.n
n_obs = env.observation_space.n
n_iterations = 1000

value_function = np.zeros(48)
base_line = 0
gamma = 0.99
Q = defaultdict(lambda: [base_line]*n_actions)
counter = defaultdict(lambda: [0]*n_actions)
policy = defaultdict(lambda: [1.0/n_actions]*n_actions)

eps = 0.8

eps_decay = 0.99
learning_curve = []

for iter_no in range(n_iterations):
    
    s = env.reset()
    
    done = False
    trajectory = []
    acc_reward = 0

    while not done:
        
        a = np.random.choice(n_actions,p=policy[s])
        
        nxt_s, r, done, _ = env.step(a)
        
        trajectory.append((s,a,r))
        
        s = nxt_s
        acc_reward += r
    
    eps *= eps_decay
    learning_curve.append(acc_reward)
        
    Gt = 0

    for i, (s,a,r) in enumerate(reversed(trajectory)):

        Gt = r + gamma*Gt
        
        if (s,a,r) not in trajectory[0:-(i+1)]:

            N_old = counter[s][a]
            Q[s][a] = (Q[s][a]*N_old + Gt)/(N_old+1)
            
            counter[s][a] += 1
            
            a_max = np.argmax(Q[s])
            
            for a in range(n_actions):
                
                if a == a_max:
                    policy[s][a] = 1.0 - eps + eps/n_actions
                else:
                    policy[s][a] = eps/n_actions
    
    print iter_no
    
# In[]

plt.figure(1)
plt.plot(learning_curve)
plt.plot(ewma(np.array(learning_curve),alpha=0.3))

# In[]
value_function = np.zeros(n_obs)

for s in range(n_obs):
    value_function[s] = np.max(Q[s])

grid = np.array(value_function).reshape(4,12)
plt.figure(1)
plt.imshow(grid)
plt.colorbar()
plt.show()
    
# In[]

states_visited = np.zeros(n_obs)

for s in range(n_obs):
    states_visited[s] = np.sum(counter[s])

grid = np.array(states_visited).reshape(4,12)
plt.figure(2)
plt.clf()
plt.imshow(grid)
plt.colorbar()
plt.show()
    
# In[]
    
done = False    
max_steps = 20
step_no  = 0

s = env.reset()

while not done and step_no < max_steps:
    
    a_max = np.argmax(policy[s])
    
    s, r, done, _ = env.step(a_max)
    
    step_no += 1
    
    env.render()
        
        
        
        
        
    
    
    
    
    



