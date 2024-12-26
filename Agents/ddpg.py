import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from framework import Agent, FeedForwardNN


class DDPGAgent(Agent):
    def __init__(self, state_size, action_size, hidden_dim=64,
                 hidden_layers=1, gamma=0.99, tau=0.005, noise=0.1,
                 batch_size=128, buffer_capacity=10000, min_experience=1):
        super().__init__(buffer_capacity)
        self.batch_size = batch_size
        self.state_size, self.action_size = state_size, action_size

        assert(min_experience >= 1)
        self.min_experience = int(batch_size * min_experience) # minimum experience required for the agent before it is allowed to learn
        self.gamma = gamma
        self.tau = tau
        self.noise = noise

        self.critic = FeedForwardNN(state_size + action_size, 1, hidden_dim, hidden_layers)
        self.critic_target = FeedForwardNN(state_size + action_size, 1, hidden_dim, hidden_layers)
        self.critic_target.eval()

        self.actor = FeedForwardNN(state_size, action_size, hidden_dim, hidden_layers, activation='tanh')
        self.actor_target = FeedForwardNN(state_size, action_size, hidden_dim, hidden_layers, activation='tanh')
        self.actor_target.eval()

        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-3)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)

        self.critic_criterion = torch.nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, state, explore=False):
        noise = np.random.normal(0, self.noise, size=self.action_size) if explore else 0.0
        noise = torch.tensor(noise, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(state) + noise
        return action
    
    def learn(self):
        if len(self) < self.min_experience:
            return None, None

        states, actions, rewards, next_states = zip(*self.sample(self.batch_size))
        state_batch = torch.cat(states).to(self.device)
        action_batch = torch.cat(actions).to(self.device)
        reward_batch = torch.cat(rewards).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
        non_final_next_state_actions = torch.cat([torch.cat([s.to(self.device), self.actor_target(s)], dim=1) for s in next_states if s is not None])

        next_q_values = torch.zeros(self.batch_size, device=self.device)
        next_q_values[non_final_mask] = self.critic_target(non_final_next_state_actions).flatten()
        target_q_values = (self.gamma * next_q_values) + reward_batch

        state_action_batch = torch.cat([state_batch, action_batch], dim=1).detach()
        q_values = self.critic(state_action_batch)
        critic_loss = self.critic_criterion(q_values, target_q_values.unsqueeze(1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(torch.cat([state_batch, self.actor(state_batch)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update the critic and actor targets by tau
        target_net_state_dict = self.critic_target.state_dict()
        critic_net_state_dict = self.critic.state_dict()
        for key in critic_net_state_dict:
            target_net_state_dict[key] = critic_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.critic_target.load_state_dict(target_net_state_dict)

        target_net_state_dict = self.actor_target.state_dict()
        actor_net_state_dict = self.actor.state_dict()
        for key in actor_net_state_dict:
            target_net_state_dict[key] = actor_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.actor_target.load_state_dict(target_net_state_dict)

        return critic_loss.item(), actor_loss.item()

        
class DDPGTrainer:
    def __init__(self, env, agent, featurizer, gamma=0.99):
        self.env = env
        self.agent = agent
        self.featurizer = featurizer
        self.gamma = gamma
        self.critic_losses = []
        self.actor_losses = []

    def train(self, episodes=1000):
        for _ in tqdm(range(episodes)):
            state, _ = self.env.reset()
            state = self.featurizer.transform_state(state)

            done = False
            while not done:
                action = self.agent.act(state, explore=True)
                
                next_state, reward, terminated, truncated, _ = self.env.step(self.featurizer.transform_action(action, self.env.action_space.low, self.env.action_space.high))

                done = terminated or truncated
                next_state = self.featurizer.transform_state(next_state) if not done else None
                reward = torch.tensor([reward], dtype=torch.float32)

                self.agent.push([state, action, reward, next_state])
                state = next_state

            critic_loss, actor_loss = self.agent.learn()
            if critic_loss is not None and actor_loss is not None:
                self.critic_losses.append(critic_loss)
                self.actor_losses.append(actor_loss)

    def run_episode(self):
        state, _ = self.env.reset()
        state = self.featurizer.transform_state(state)
        frames = []
        actions = []
        total_reward = 0.0

        done = False
        while not done:
            action = self.agent.act(state, explore=False)
            a = self.featurizer.transform_action(action, self.env.action_space.low, self.env.action_space.high)
            actions.append(a)
            next_state, reward, terminated, truncated, _ = self.env.step(a)

            done = terminated or truncated
            total_reward += reward
            state = self.featurizer.transform_state(next_state)
            frames.append(self.env.render())
        
        return {"reward": total_reward, "steps": len(frames), "rgb_arrays": frames, "actions": actions}
                

    def plot_losses(self):
        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(np.convolve(self.critic_losses, np.ones(100)/100, mode='valid'))
        ax[0].set_title("Critic Loss")
        ax[1].plot(np.convolve(self.actor_losses, np.ones(100)/100, mode='valid'))
        ax[1].set_title("Actor Loss")
        plt.show()