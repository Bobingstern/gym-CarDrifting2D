# Top Down Car Driving for Gym
<h2>A custom gym environment for a top down drifting game</h2>

<p align="left">
  <img width="50%" src="https://raw.githubusercontent.com/Bobingstern/gym-CarDrifting2D/main/images/Main.png"></img>
  <!--<br/><i>State-space observation mode.</i>-->
</p>

<h3>Requirements:</h3>
Python 3.8+

<h3>Install using the pip package:</h3> 

```
pip install gym-CarDrifting2D
```

<h4>Here is an example with random actions:</h4>

```python
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
```
<h4>The agent has a total of 9 actions which are:</h4>

1) Forward

2) Backward

3) Left

4) Right

5) Forward Left

6) Forward Right

7) Backward Left

8) Backward Right

9) Do nothing

There are numerous hidden Reward Gates around the track that the agent must cross to get a reward

The agent will receive a constant negative reward of -0.01 unless it either crosses a reward gate which will result in +1 reward or crashes into the wall which will result a -1 reward and cause the ```done``` flag to become ```True``` indicating that you should call ```env.reset()```

<h2>For those interested in the game Physics!</h2>

The agent has a vector called ```pos``` which stores the car's location, a vector called ```vel``` which stores the car's velocity along with a variable called ```drag``` which is it's drag. It has a variable called ```power``` which is how fast the car accelerates and 3 more variables which are ```angularVel``` which is the Angular Velocity ```angularDrag``` which is the Angular Drag and ```turnSpeed``` which is the turning speed. It also has one more variable called ```angle``` which is the angle of the car affected by ```angularVel```.
```python
self.pos = [650, 200]
self.velX = 0
self.velY = 0
self.drag = drag
self.angularVel = 0.0
self.angularDrag = angularDrag
self.power = power
self.turnSpeed = turnSpeed
self.angle = math.radians(-90)
```
Each step the following operations are executed
```python
self.pos[0] += self.velX
self.pos[1] += self.velY

self.velX *= self.drag
self.velY *= self.drag
self.angle += self.angularVel
self.angularVel *= self.angularDrag
```
And here are the control functions:
```python
def acc(self):
    self.velX += math.sin(self.angle) * self.power
    self.velY += math.cos(self.angle) * self.power

    if (self.velX > 10):
        self.velX = 10

    if (self.velY > 10):
        self.velY = 10

def decc(self):
    self.velX -= math.sin(self.angle) * self.power
    self.velY -= math.cos(self.angle) * self.power

    if (self.velX < -10):
        self.velX = -10

    if (self.velY < -10):
        self.velY = -10

def right(self):
    self.angularVel += self.turnSpeed

def left(self):
    self.angularVel -= self.turnSpeed
```
<h2>Enjoy! And don't forget to leave feedback on improvements or bugs!</h2>
