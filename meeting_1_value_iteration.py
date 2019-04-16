#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:39:23 2019

@author: x
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v0")

n_actions = env.action_space.n
value_function = np.zeros(48)

gamma = 0.99
P = env.P
delta = 100000000
threshold = 0.1

while delta>threshold:
    
    delta = 0
    
    for s in P:
        
        P_s = P[s]
        v_max = -100000000
        v_old = value_function[s]
        
        for a in P_s:
            
            nxt_s, r, d = P_s[a][0][1:]
            
            v_nxt_state = value_function[nxt_s]*(1-d)
            
            v = r + gamma * v_nxt_state
            
            v_max = max(v,v_max)
            
        delta = max(delta,abs(v_old - v_max))
        value_function[s] = v_max
    
grid = np.array(value_function).reshape(4,12)

plt.imshow(grid)
plt.colorbar()
plt.show()

s = env.reset()
done = False

while not done:

    
    P_s = P[s]
    v_max = -100000000
    a_max = 0
    
    for a in P_s:
        
        nxt_s = P_s[a][0][1]
        d = P_s[a][0][3]
        r = P_s[a][0][2]
        value_nxt_state = value_function[nxt_s]
        
        
        v = r + gamma * value_nxt_state*(1-d)
        
        if v > v_max:
            v_max = v
            a_max = a
  
    s, r_, done, p_ = env.step(a_max)
    
    env.render()
    