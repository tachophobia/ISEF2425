import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from framework import Agent, FeedForwardNN

sys.path.append(f"{(p:=sys.path[0])[:p.index('Agents')]}Lichtenberg Optimization")
from la import LichtenbergAlgorithm
from func import ObjectiveFunction


class DDPGAgent(Agent):
    def __init__(self, state_size, action_size, hidden_dim=64,
                 hidden_layers=1, gamma=0.99, tau=0.005, noise=0.1,
                 batch_size=128, alpha=1e-3, buffer_capacity=10000):
        super().__init__(buffer_capacity)
        self.batch_size = batch_size
        self.state_size, self.action_size = state_size, action_size

        self.gamma = gamma
        self.tau = tau
        self.noise = noise

        self.critic = FeedForwardNN(state_size + action_size, 1, hidden_dim, hidden_layers)
        self.critic_target = FeedForwardNN(state_size + action_size, 1, hidden_dim, hidden_layers)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor = FeedForwardNN(state_size, action_size, hidden_dim, hidden_layers, activation='tanh')
        self.actor_target = FeedForwardNN(state_size, action_size, hidden_dim, hidden_layers, activation='tanh')
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=alpha)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=alpha)

        self.critic_criterion = torch.nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, state, explore=False):
        noise = np.random.normal(0, self.noise, size=self.action_size) if explore else 0.0
        noise = torch.tensor(noise, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(state) + noise
        return action
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states = zip(*self.sample(self.batch_size))
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
        non_final_next_state_actions = torch.cat([torch.cat([s.to(self.device), self.actor_target(s)], dim=1) for s in next_states if s is not None])

        next_q_values = torch.zeros(self.batch_size, device=self.device)
        next_q_values[non_final_mask] = self.critic_target(non_final_next_state_actions).flatten()
        target_q_values = (self.gamma * next_q_values) + rewards

        state_actions = torch.cat([states, actions], dim=1).detach()
        q_values = self.critic(state_actions)
        critic_loss = self.critic_criterion(q_values, target_q_values.unsqueeze(1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(torch.cat([states, self.actor(states)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update_target_nets()

        return critic_loss.item(), actor_loss.item()
    
    def soft_update_target_nets(self):
        target = self.critic_target.state_dict()
        current = self.critic.state_dict()
        for key in current:
            target[key] = current[key] * self.tau + target[key] * (1 - self.tau)
        self.critic_target.load_state_dict(target)

        target = self.actor_target.state_dict()
        current = self.actor.state_dict()
        for key in current:
            target[key] = current[key] * self.tau + target[key] * (1 - self.tau)
        self.actor_target.load_state_dict(target)


class ActionValueFunction(ObjectiveFunction):
    def __init__(self, net, dim=2):
        self.net = net
        self.state = None
        self.lower_bound = np.array([-1] * dim)
        self.upper_bound = np.array([1] * dim)
        self.trigger = np.array([0] * dim)

    def bounds(self):
        return self.lower_bound, self.upper_bound
    
    def center(self):
        return self.trigger

    def evaluate(self, X):
        action = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        state_action = torch.cat([self.state, action], dim=1)
        return -self.net(state_action).item()


class LichtenbergAgent(DDPGAgent):
    def __init__(self, state_size, action_size, figure_path, hidden_dim=64,
                  hidden_layers=1, gamma=0.99, tau=0.005, noise=0.1,
                 batch_size=128, alpha=1e-3, buffer_capacity=10000):
        super().__init__(state_size=state_size,
                         action_size=action_size,
                         hidden_dim=hidden_dim,
                         hidden_layers=hidden_layers,
                         gamma=gamma, 
                         tau=tau, 
                         noise=noise,
                         batch_size=batch_size,
                         alpha=alpha,
                         buffer_capacity=buffer_capacity)
        
        self.la = LichtenbergAlgorithm(M=2, filename=figure_path)
        self.q_approximator = ActionValueFunction(self.critic, action_size)

    def act(self, state, explore=False):
        if explore:
            self.q_approximator.state = state
            self.q_approximator.trigger = np.random.normal(0, self.noise, size=self.action_size)
            action = self.la.optimize(self.q_approximator, n_iter=3, pop=30)
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
            return action
        return self.actor(state)


class DDPGTrainer:
    def __init__(self, env, agent, featurizer, gamma=0.99, **kwargs):
        self.env = env
        self.agent = agent
        self.featurizer = featurizer
        self.gamma = gamma
        self.episode_rewards = []
        self.critic_losses = []
        self.actor_losses = []

        defaults = {"until_convergence": False}
        kwargs = {**defaults, **kwargs}
        self.check_convergence = kwargs['until_convergence']
        if self.check_convergence:
            assert "convergence_reward" in kwargs, "Convergence reward must be provided"
            self.convergence_reward = kwargs['convergence_reward']

    def train(self, episodes=1000):
        for e in tqdm(range(episodes)):
            state, _ = self.env.reset()
            state = self.featurizer.transform_state(state)

            done = False
            total_reward = 0.0
            while not done:
                action = self.agent.act(state, explore=True)
                
                next_state, reward, terminated, truncated, _ = self.env.step(self.featurizer.transform_action(action, self.env.action_space.low, self.env.action_space.high))

                done = terminated or truncated
                next_state = self.featurizer.transform_state(next_state) if not done else None
                total_reward += reward
                reward = torch.tensor([reward], dtype=torch.float32)

                self.agent.push([state, action, reward, next_state])
                state = next_state

            critic_loss, actor_loss = self.agent.learn()
            if critic_loss is not None and actor_loss is not None:
                self.critic_losses.append(critic_loss)
                self.actor_losses.append(actor_loss)
                self.episode_rewards.append(total_reward)
            
            if e % 100 == 0 and self.has_converged():
                print(f"Converged at episode {e}")
                print(self.has_converged())
                break
    
    def has_converged(self, episodes=100):
        if self.check_convergence:
            rewards = []
            for _ in range(episodes): # run several episodes in case environment isn't deterministic
                rewards.append(self.run_episode(render=False)['reward'])
            return bool(np.mean(rewards) >= self.convergence_reward)
        return False

    def run_episode(self, render=True):
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
            if render:
                frames.append(self.env.render())
        
        return {"reward": total_reward, "steps": len(frames), "rgb_arrays": frames, "actions": actions}
                

    def plot_losses(self):
        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(np.convolve(self.critic_losses, np.ones(100)/100, mode='valid'))
        ax[0].set_title("Critic Loss")
        ax[1].plot(np.convolve(self.actor_losses, np.ones(100)/100, mode='valid'))
        ax[1].set_title("Actor Loss")
        plt.show()

    def plot_rewards(self):
        plt.plot(np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid'))
        plt.title("Episode Rewards")
        plt.show()

    def save(self, path='trainer.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)