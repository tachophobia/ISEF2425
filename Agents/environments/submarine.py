import numpy as np
import matplotlib.pyplot as plt
import io
import gym

from matplotlib.patches import Rectangle
from gym import spaces

class ContinuousSubmarine(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        
        # Set default parameters and update with kwargs
        defaults = {
            'width': 50,
            'height': 55,
            'start_pos': (1, 1),
            'terminal_width': 5,
            'terminal_height': 5,
            'max_y': {0: 10, 5: 15, 10: 20, 15: 25, 20: 25, 25: 25, 30: 40, 35: 40, 40: 50, 45: 55},
            'rewards': {(5, 0): 1, (10, 5): 2, (15, 10): 3, (20, 15): 5, (20, 20): 8, (20, 25): 16, (35, 30): 24, (35, 35): 50, (45, 40): 74, (50, 45): 124},
            'max_timesteps': 500,
            'delta_t': 0.1,
            'drag_coefficient': 0.05
        }
        params = {**defaults, **kwargs}

        # Environment dimensions and terminal areas
        self.width = params['width']
        self.height = params['height']
        self.terminal_width = params['terminal_width']
        self.terminal_height = params['terminal_height']

        # Initial state: [y, x, vy, vx]
        self.initial_state = [*params['start_pos'], 0, 0]
        self.state = self.initial_state

        # Seafloor boundary represented as {x: max_y}
        self.max_y = params['max_y']

        # Rewards for terminal areas
        self.rewards = params['rewards']

        # Action space: [theta (0 to 2*pi), magnitude (0.05 to 1.5)]
        self.action_space = spaces.Box(
            low=np.array([0, 0.05]),
            high=np.array([2 * np.pi, 1.5]),
            dtype=np.float64
        )

        # Observation space: [y, x, vy, vx] bounded by environment dimensions and reasonable velocities
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf]),
            high=np.array([self.height, self.width, np.inf, np.inf]),
            dtype=np.float64
        )

        self.max_timesteps = params['max_timesteps']
        self.timestep = 0
        self.delta_t = params['delta_t']

        self.drag_coefficient = params['drag_coefficient']

    def reset(self):
        """Reset the environment to the initial state."""
        self.state = self.initial_state
        self.timestep = 0
        return np.array(self.state, dtype=np.float32), {} # return the state and info

    def step(self, action):
        """Perform one step in the environment based on the given action."""
        assert self.action_space.contains(action), "Invalid action."
        
        delta_x, delta_y, vx, vy = self._calculate_dynamics(action)
        self.state[0] += delta_y
        self.state[1] += delta_x
        self.state[2] = vy
        self.state[3] = vx
        self.timestep += 1
        
        box = self._to_box(self.state[:2])

        bounded = 0 <= self.state[0] <= self.height and 0 <= self.state[1] <= self.width
        terminal = box in self.rewards or not bounded
        reward = self.rewards.get(box, -1) # penalize the agent for each timestep spent

        truncated = self.timestep > self.max_timesteps
        done = terminal or truncated
        return np.array(self.state, dtype=np.float32), reward, done, truncated, {}

    def _to_box(self, pos):
        """Convert position to the top-left corner of the larger box."""
        y, x = int(pos[0]), int(pos[1])
        return (y // self.terminal_height) * self.terminal_height, (x // self.terminal_width) * self.terminal_width

    def _calculate_dynamics(self, action):
        """Calculate the new state dynamics given an action."""
        theta, magnitude = action
        x_accel = magnitude * np.cos(theta)
        y_accel = magnitude * np.sin(theta)

        # Calculate drag forces
        drag_x = -self.drag_coefficient * self.state[3]
        drag_y = -self.drag_coefficient * self.state[2]
        x_accel += drag_x
        y_accel += drag_y

        # Calculate displacement
        delta_x = self.state[3] * self.delta_t + 0.5 * x_accel * (self.delta_t ** 2)
        delta_y = self.state[2] * self.delta_t + 0.5 * y_accel * (self.delta_t ** 2)

        # Update velocities
        vx = self.state[3] + self.delta_t * x_accel
        vy = self.state[2] + self.delta_t * y_accel

        return delta_x, delta_y, vx, vy

    def is_valid_action(self, action):
        """Check if the given action is valid in the current state."""
        theta, magnitude = action
        if not (0 <= theta <= 2 * np.pi and 0.05 <= magnitude <= 1.5):
            return False
        
        delta_x, delta_y, _, _ = self._calculate_dynamics(action)
        next_x = self.state[1] + delta_x
        next_y = self.state[0] + delta_y
        if next_x < 0 or next_x > self.width or next_y < 0:
            return False
        next_box = self._to_box([next_y, next_x])
        return next_y <= self.max_y.get(next_box[1], self.height)
    
    def render(self, mode="human"):
        """Render the environment based on the mode."""
        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            if mode != 'human':
                plt.close(self.fig)

        self.ax.clear()

        # Draw the seafloor boundaries
        for x in range(0, self.width + 1, self.terminal_width):
            max_y = self.max_y.get(x, self.height)
            self.ax.add_patch(Rectangle((x, max_y), self.terminal_width, self.height - max_y, color='brown', alpha=0.6))

        # Draw the reward regions
        for (y, x), reward in self.rewards.items():
            self.ax.add_patch(Rectangle((x, y), self.terminal_width, self.terminal_height, color='green', alpha=0.5))
            self.ax.text(x + self.terminal_width / 2, y + self.terminal_height / 2, f"{reward}", color='black', ha='center', va='center')

        y, x, vy, vx = self.state
        self.ax.plot(x, y, 'ro', markersize=10, label='Submarine')
        self.ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, color='black', label='Velocity', width=0.005)


        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.invert_yaxis() 
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Submarine Environment")
        self.ax.text(0.85, 0.95, f"Timestep: {self.timestep}", transform=self.ax.transAxes)
        self.ax.grid(True)
        if mode == "rgb_array":
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png")
            buf.seek(0)
            image = plt.imread(buf)
            buf.close()
            return image
        elif mode == 'human':
            plt.pause(0.05)


if __name__ == "__main__":
    env = ContinuousSubmarine()
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, _, _ = env.step(action)
        env.render(mode='human')
      