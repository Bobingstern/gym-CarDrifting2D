import gym
import gym_Drifting2D
import random


env = gym.make("CarDrifting2D-v0", drag=0.9, power=0.7, turnSpeed=0.04, angularDrag=0.6) 

#Parameter Definitions:
#Drag, how much the car skids, the higher the more skid
#power, how fast the car accelerates
#turnSpeed, how fast the car turns
#angularDrag, how much the car spins, the higher the more spin

env.reset()

num_states = env.states
num_actions = env.actions
while True:
    action = random.randint(0, num_actions)
    state, reward, done, _ = env.step(action)
    if (done):
        env.reset()
    env.render()
