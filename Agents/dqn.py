import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from framework import Agent, FeedForwardNN
from tqdm import tqdm
from itertools import count
from IPython import display

class DQNAgent(Agent):
    def __init__(self, input_dim, output_dim, hidden_dim=64,
                 hidden_layers=5, gamma=0.99, tau=0.005,
                 min_epsilon=0.1, epsilon_decay=0.999,
                 batch_size=32, buffer_capacity=10000):
        super().__init__(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = 1.0
        self.decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.action_dim = output_dim

        self.policy_net = FeedForwardNN(input_dim, output_dim, hidden_dim, hidden_layers)
        self.target_net = FeedForwardNN(input_dim, output_dim, hidden_dim, hidden_layers)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-3, amsgrad=True)
        self.criterion = torch.nn.SmoothL1Loss()
        self.device = self.target_net.device
    
    def select_action(self, state, explore=False):
        """
        Select the next action either from existing policy or random.
        
        Args:
            state ([DATA TYPE]): the current state of the agent.
            explore (bool): whether exploration is allowed.

        Returns:
            [DATA TYPE]: the selected action.
        """
        if explore and random.random() < self.epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                action = self.policy_net(state).max(1).indices.view(1, 1)
        return action
    
    def step(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
    
    def update(self):
        if len(self) < self.batch_size:
            return
        
        # update the weights based on the agent's experience in buffer
        states, actions, rewards, next_states = zip(*self.sample(self.batch_size))

        self.policy_net.train()

        state_batch = torch.cat(states).to(self.device)
        action_batch = torch.cat(actions).to(self.device)
        reward_batch = torch.cat(rewards).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None]).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device) # no future reward if final
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        target_q_values = (self.gamma * next_state_values) + reward_batch

        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        return loss.item()
    
    def load(self, filepath):
        self.policy_net.load(filepath)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    
class DQNTrainer:
    def __init__(self, env, agent, featurizer,  batch_size=256, gamma=0.99):
        self.env = env
        self.agent = agent
        self.featurizer = featurizer
        self.batch_size = batch_size
        self.gamma = gamma
        self.losses = []
        self.durations = []

    def run_episode(self, render=False):
        state, _ = self.env.reset()
        state = self.featurizer.transform_state(state)

        done = False
        total_reward = 0.
        frames = []
        while not done:
            action = self.agent.select_action(state, explore=False)

            next_state, reward, terminated, truncated, _ = self.env.step(self.env.action_space[action])
            done = terminated or truncated
            next_state = self.featurizer.transform_state(next_state)

            state = next_state
            total_reward += reward

            img = self.env.render()
            frames.append(img)
            if render and 'inline' in matplotlib.get_backend():  # ipython rendering
                plt.imshow(img)
                plt.axis('off')
                display.display(plt.gcf())
                display.clear_output(wait=True)

        return {"reward": total_reward, "steps": len(frames), "rgb_arrays": frames}

    def train(self, episodes=1000):
        for _ in tqdm(range(episodes)):
            state, _ = self.env.reset()
            state = self.featurizer.transform_state(state)

            done = False
            for t in count():
                action = self.agent.select_action(state, explore=True)

                next_state, reward, terminated, truncated, _ = self.env.step(self.env.action_space[action])
                done = terminated or truncated

                next_state = self.featurizer.transform_state(next_state) if not done else None
                reward = torch.tensor([reward], dtype=torch.float32)

                self.agent.push([state, action, reward, next_state])

                state = next_state

                loss = self.agent.update()
                if loss is not None:
                    self.losses.append(loss)
                self.agent.step()

                if done:
                    self.durations.append(t + 1)
                    break
    
    def plot_losses(self):
        losses = np.array(self.losses)
        losses = np.convolve(losses, np.ones(100)/100, mode='valid')
        plt.plot(losses)
        plt.title("Loss during training")
        plt.show()
    
    def plot_durations(self):
        plt.plot(self.durations)
        plt.title("Episode durations")
        plt.show()