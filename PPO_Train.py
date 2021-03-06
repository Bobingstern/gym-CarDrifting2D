import gym
import gym_Drifting2D
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
from PPO import *
import matplotlib.pyplot as plt

def main():
    ############## Hyperparameters ##############
    env_name = "CarDrifting2D-v0"
    # creating environment
    env = gym.make(env_name, drag=0.93)
    state_dim = env.states
    action_dim = env.actions
    render = True
    solved_reward = 1000  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 50000  # max training episodes
    max_timesteps = 5000  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 100  # update policy every n timesteps
    lr = 5e-4
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    rewards = []
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()

        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        name = "Data/"+env_name
        f = open(name, "a+")
        f.write(str(running_reward)+"\n")
        f.close()

        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name+"Solved"))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            torch.save(ppo.policy.state_dict(), './SavedModels/PPO_{}.pth'.format(env_name))
            running_reward = 0
            avg_length = 0



if __name__ == '__main__':
    main()

