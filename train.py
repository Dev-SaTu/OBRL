import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PPO:
    def __init__(self, policy_net, value_net, lr=0.001, gamma=0.99, eps_clip=0.2):
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def update(self, old_probs, states, actions, rewards, masks):
        returns = []
        discounted_reward = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            discounted_reward = reward + self.gamma * discounted_reward * mask
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)
        values = self.value_net(states).squeeze(1) # squeeze 추가함
        advantages = returns - values.detach()

        # PPO update
        for _ in range(1): # 10 에서 1 로..
            new_probs = self.policy_net(states).gather(1, actions)
            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(returns, values)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()


if __name__ == "__main__":

    # 환경 초기화
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 네트워크 및 PPO 객체 초기화
    policy_net = PolicyNetwork(state_dim, 128, action_dim)
    value_net = ValueNetwork(state_dim, 128)
    ppo_agent = PPO(policy_net, value_net)

    # Hyperparameters
    num_epochs = 500
    num_timesteps = 200

    for epoch in range(num_epochs):
        state = env.reset()
        old_probs = []
        states = []
        actions = []
        rewards = []
        masks = []
        total_reward = 0

        for t in range(num_timesteps):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            action = np.random.choice(action_dim, p=action_probs.detach().numpy())
            next_state, reward, done, _ = env.step(action)

            old_probs.append(action_probs[action].item())
            states.append(state_tensor) 
            actions.append(action)
            rewards.append(reward)
            masks.append(0 if done else 1)
            
            total_reward += reward
            state = next_state

            if done:
                break

        # 경험을 사용하여 PPO 업데이트
        old_probs_tensor = torch.tensor(old_probs)
        states_tensor = torch.stack(states)
        actions_tensor = torch.tensor(actions).unsqueeze(1)
        ppo_agent.update(old_probs_tensor, states_tensor, actions_tensor, rewards, masks)

        print(f"Epoch {epoch}, Total Reward: {total_reward}")

    env.close()