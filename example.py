import gym
import gym_Drifting2D
import random

env = gym.make("CarDrifting2D-v0", drag=0.9, power=1, turnSpeed=0.04, angularDrag=0.6, multiInputs=False, showGates=False, constantAccel=False)
# Parameter Definitions:
# Drag, how much the car skids, the higher the more skid
# power, how fast the car accelerates
# turnSpeed, how fast the car turns
# angularDrag, how much the car spins, the higher the more spin
# Multi Inputs means the agent can go both forward/backward AND left or right simultaneously
# Show Gates is to show the reward gates
# constant accel is to accelerate constantly
env.reset()

num_states = env.states
num_actions = env.actions

while True:
    action = random.randint(0, num_actions)
    state, reward, done, _ = env.step(action)
    if done:
        env.reset()
    env.render()
