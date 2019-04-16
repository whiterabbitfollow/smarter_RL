import gym

env = gym.make("CliffWalking-v0")

# In[]

s = env.reset()

for i in range(10):


    a = env.action_space.sample()
    
    nxt_s, r, done, p = env.step(a)
    
    print nxt_s, r, done
    env.render()
