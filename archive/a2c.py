import os
import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error, SparseCategoricalCrossentropy, CategoricalCrossentropy
from src.models import a2c_Model

class A2CAgent:
    def __init__(self, model, lr=5e-3, gamma=0.95, value_c=0.5, entropy_c=1e-4):
        #hyperparameters for the training process
        self.gamma = gamma
        self.lr = lr
        self.value_c = value_c
        self.entropy_c = entropy_c

        #loading a2c model
        self.model = model
        #we compile the network with 2 custom losses, one for the logits(actor) and one for the value function(critic)
        self.model.network.compile(optimizer=RMSprop(lr=self.lr), loss=[self.logits_loss, self.valuefn_loss])

    def train(self, env, batch_size=64, updates=250):
        #defining storage for all the variables
        actions = np.empty((batch_size, ), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,)+env.observation_space.shape)

        ep_rewards = [0.0]
        next_obs = env.reset()
        env.render()
        for update in range(updates):
            for step in range(batch_size):
                observations[step]=next_obs.copy()
                actions[step], values[step] = self.model.action_value(np.expand_dims(observations[step,:], axis=0))
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                env.render()

                ep_rewards[-1]+=rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    env.render()
                    print("Episode: %03d || Reward: %03d" % (len(ep_rewards)-1,ep_rewards[-2]))

            _, next_value = self.model.action_value(next_obs[None,:])  #finding the estimated value function of the last observation
            returns, advs = self.return_advantages(rewards, dones, values, next_value) #finding the estimated returns and the advantages
            acts_and_advs = np.concatenate([actions[:,None], advs[:,None]], axis=-1) #passing actions and advantages together for the loss function
            losses=self.model.network.train_on_batch(observations, [acts_and_advs, returns])
            print("[%d/%d] Losses: %s" % (update+1, update, losses))
            self.model.network.save('models/model'+str(update)+'.h5')

        return ep_rewards

    def test_model(self, env, network, episodes):
        self.model.network = load_model(network, custom_objects={'logits_loss':self.logits_loss, 'valuefn_loss':self.valuefn_loss})
        episode_rewards=[]
        for episode in range(episodes):
            next_obs = env.reset()
            env.render()
            done=False
            episode_rewards.append(0.0)
            while not done:
                action,value = self.model.action_value(np.expand_dims(next_obs, axis=0))
                next_obs, reward, done, _ = env.step(action)
                env.render()
                episode_rewards[-1]+=reward
                if done:
                    episode_rewards.append(0.0)
                    print("Episode Rewards: "+str(episode_rewards[-2]))
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(episode_rewards), 10), episode_rewards[::10])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
     

    def return_advantages(self, rewards, dones, values, next_value):
        #function to return the advantages for all the timesteps
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(rewards.shape[0])):
            if t>61:
                continue
            #loop to find the return for all the timesteps
            #the value function estimate for all the timsesteps is made using MC value function estimate
            #however, if the last observation is not the end of the episode then there is some break
            returns[t] = rewards[t] + self.gamma*returns[t+1]*(1-dones[t])
        returns = returns[:-1]
        #subtracting baseline to reduce variance
        advantages = returns - values
        return returns, advantages

    def valuefn_loss(self, returns, value):
        #custom valuefn_loss
        #defined as mse between the estimated return and the valuefn predicted by the network
        #used to update value function network parameters
        return self.value_c * mean_squared_error(returns, value)

    def logits_loss(self, actions_and_advantages, logits):
        #function to calculate the policy gradients
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        #sparse categorical crossentropy can be used to calculate loss between stochastic policy predictions and the true action taken 
        #instead of taking gradients of log probabilites of the predictions, I've calculated the loss between the policy and the true action directly
        weighted_sparse_ce = SparseCategoricalCrossentropy(from_logits=True) 
        actions = tf.cast(actions, tf.int32)
        #we also give advantages as sample_weights 
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        probs = tf.nn.softmax(logits)
        cce = CategoricalCrossentropy()
        entropy_loss = cce(probs, probs)
        return policy_loss - self.entropy_c * entropy_loss

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.render()
    model = a2c_Model(env.observation_space.shape[0],env.action_space.n)
    agent = A2CAgent(model)

    rewards_history = agent.train(env, 64, 1000)
    print("Finished training. Testing...")
    agent.test_model(env, 'models/model999.h5', 100)

    plt.style.use('seaborn')
    plt.plot(np.arange(0, len(rewards_history), 10), rewards_history[::10])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()