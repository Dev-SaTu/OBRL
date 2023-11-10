import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical
from my_env import OBRLEnv

def get_gae(rewards: list, values: list, is_terminals: list, gamma: float, lamda: float):
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = (rewards[i] + gamma * values[i + 1] * is_terminals[i] - values[i])
        gae = delta + gamma * lamda * is_terminals[i] * gae
        returns.insert(0, gae + values[i])
    return returns

def trajectories_data_generator(states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, log_probs: torch.Tensor, values: torch.Tensor, advantages: torch.Tensor, batch_size, num_epochs):
    data_len = states.size(0)
    for _ in range(num_epochs):
        for _ in range(data_len // batch_size):
            ids = np.random.choice(data_len, batch_size)
            yield states[ids, :], actions[ids], returns[ids], log_probs[ids], values[ids], advantages[ids]

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Actor(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Actor, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden1(state))
        x = self.softmax(self.out(x))
        probs = torch.softmax(x, dim=1) # a_actor_critic.py
        dist = Categorical(probs) # a_actor_critic.py
        action = dist.sample() # a_actor_critic.py
        return action, dist # a_actor_critic.py

class Critic(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super(Critic, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden(state))
        value = self.out(x)
        return value
    
class Memory:
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []

class PPOAgent(object):

    def __init__(self, env: OBRLEnv):
        self.is_evaluate = False
        self.lamda = 0.95
        self.entropy_coef = 0.05
        self.value_range = 0.5
        self.rollout_len = 500
        self.total_rollouts = 5000
        self.num_epochs = 12
        self.batch_size = 50
        self.scores = []
        self.solved_reward = 10000
        
        
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 0.2
        self.env = env
        self.memory = Memory()
        self.device = 'cpu'
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.actor = Actor(input_dim=self.obs_dim, hidden_dim=128, output_dim=self.act_dim).apply(init_weights).to(self.device) # b_ppo
        self.critic = Critic(input_dim=self.obs_dim, hidden_dim=128).apply(init_weights).to(self.device) # b_ppo
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
    def get_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)

        if not self.is_evaluate:
            value = self.critic(state)
           
            self.memory.states.append(state)
            self.memory.actions.append(action)
            self.memory.log_probs.append(dist.log_prob(action))
            self.memory.values.append(value)

        return list(action.detach().cpu().numpy()).pop()
    
    def step(self, action: int):
        next_state, reward, done = self.env.step(action=action)

        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        self.memory.rewards.append(torch.FloatTensor(reward).to(self.device))
        self.memory.is_terminals.append(torch.FloatTensor(1 - done).to(self.device))
        return next_state, reward, done

    def train(self):
        total_train_start_time = time.time()

        score = 0
        state = self.env.reset()
        state = np.reshape(state, (1, -1))
        num_episode = 0
        time_step = 0
        episode_reward_list = []
        episode_reward = 0

        for step_ in range(self.total_rollouts):
            for _ in range(self.rollout_len):
                action = self.get_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]
                episode_reward += reward[0][0]

                if done[0][0]:
                    self.scores.append(score)
                    score = 0
                    state = self.env.reset()
                    state = np.reshape(state, (1, -1))
                    num_episode += 1
                    episode_reward_list.append(episode_reward)
                    episode_reward = 0

                time_step += 1

                if done[0][0]:
                    total_training_time = time.time() - total_train_start_time
                    total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
                    print("[Episode {:3,}, Steps {:6,}]".format(num_episode, time_step), "Episode Reward: {:>9.3f},".format(np.mean(episode_reward_list)), "Elapsed Time: {}".format(total_training_time))
                    
            if self.solved_reward is not None:
                if np.mean(self.scores[-10:]) > self.solved_reward:
                    print("It's solved!")
                    break

            value = self.critic(torch.FloatTensor(next_state))
            self.memory.values.append(value)
            self.update()

        self.save()
        self.env.close()

    def update(self):
        returns = get_gae(self.memory.rewards, self.memory.values, self.memory.is_terminals, self.gamma, self.lamda)
        actor_losses, critic_losses = [], []

        states = torch.cat(self.memory.states).view(-1, self.obs_dim)
        actions = torch.cat(self.memory.actions)
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(self.memory.log_probs).detach()
        values = torch.cat(self.memory.values).detach()
        print('shape: ', returns.shape, values.shape)
        advantages = returns - values[:-1]

        for state, action, return_, old_log_prob, old_value, advantage in trajectories_data_generator(states=states, actions=actions, returns=returns, log_probs=log_probs, values=values, advantages=advantages, batch_size=self.batch_size, num_epochs=self.num_epochs):

            action, dist = self.actor(state)
            cur_log_prob = dist.log_prob(action)
            ratio = torch.exp(cur_log_prob - old_log_prob)

            entropy = dist.entropy().mean()

            loss = advantage * ratio
            clipped_loss = (torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * advantage)
            actor_loss = (-torch.mean(torch.min(loss, clipped_loss)) - entropy * self.entropy_coef)
            
            cur_value = self.critic(state)

            critic_loss = (return_ - cur_value).pow(2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

    def evaluate(self):
        state = self.env.reset()
        state = np.reshape(state, (1, -1))
        done = False

        for _ in range(self.rollout_len):
            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                state = np.reshape(state, (1, -1))

        self.env.close()

    def save(self):
        torch.save(self.actor.state_dict(), 'actor.pth')
        torch.save(self.critic.state_dict(), 'critic.pth')

    def load(self):

        self.actor.load_state_dict(torch.load('actor.pth'))
        self.critic.load_state_dict(torch.load('critic.pth'))

if __name__ == "__main__":
    seed = 555

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = OBRLEnv()
    ppo_agent = PPOAgent(env)
    ppo_agent.train()