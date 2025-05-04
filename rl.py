import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import random


#----------------------Environment----------------------#

class ChallengingMazeEnv:
    def __init__(self, size=12):
        self.size = size
        self.max_steps = 100
        self.action_space = 4
        self.observation_space = size * size
        
        self.maze = np.zeros((size, size))
        self.maze[1:size-1, 3] = 1
        self.maze[2:size-2, 6] = 1
        self.maze[3:size-3, 9] = 1
        self.maze[4, 1:8] = 1
        self.maze[7, 3:10] = 1
        self.maze[7,0] = 1
        self.maze[6,1] = 1
        self.maze[0,4] = 1
        self.maze[size-4, 2:size-2] = 1
        self.maze[5:8, 5] = 1
        self.maze[2, 4:7] = 1
        self.maze[8, 7:11] = 1
        
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.current_pos = self.start_pos
        self.steps = 0
        
        self.cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue', 'yellow'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        self.best_path = None
        self.best_steps = float('inf')
        self.visited = set()
        self.current_path = []
        
    def reset(self):
        self.current_pos = self.start_pos
        self.steps = 0
        self.visited = set()
        self.current_path = []
        return self._get_state()
    
    def _get_state(self):
        return self.current_pos[0] * self.size + self.current_pos[1]
    
    def step(self, action):
        x, y = self.current_pos
        
        if action == 0: x = max(x - 1, 0)
        elif action == 1: x = min(x + 1, self.size - 1)
        elif action == 2: y = max(y - 1, 0)
        elif action == 3: y = min(y + 1, self.size - 1)
            
        if self.maze[x, y] == 1:
            return self._get_state(), -15, True
        
        self.current_pos = (x, y)
        self.visited.add((x, y))
        self.current_path.append((x, y))
        self.steps += 1
        
        if self.current_pos == self.goal_pos:
            if self.steps < self.best_steps:
                self.best_steps = self.steps
                self.best_path = self.current_path.copy()
            return self._get_state(), 20, True
        
        if self.steps >= self.max_steps:
            return self._get_state(), -10, True  # New penalty for max steps reached 

        return self._get_state(), -2, False
    
    def get_visualization(self):
        grid = self.maze.copy()
        grid[self.current_pos] = 2  # Current position (green)
        grid[self.goal_pos] = 3     # Goal (red)
        
        # Mark all visited positions in blue
        for pos in self.visited:
            if grid[pos] == 0:  # Only mark if not a wall/goal/current
                grid[pos] = 4   # Blue (visited)
        
        # Mark best path in yellow
        if self.best_path:
            for pos in self.best_path:
                if grid[pos] == 4:  # Only overwrite visited positions
                    grid[pos] = 5   # Yellow (best path)
        
        return grid

#----------------------Agent----------------------#

class ImperfectQLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration
        self.q_table = np.zeros((env.observation_space, env.action_space))
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space - 1)  # Exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation (shortest path)
    
    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



def animate_episodes(env, agent, num_episodes=5, max_steps=100):
    fig, ax = plt.subplots(figsize=(8, 8))
    episodes = []

    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Turn off exploration for evaluation

    for episode in range(num_episodes):
        state = env.reset()
        env.visited = set()
        done = False
        frames = []
        steps = 0
        result = "IN PROGRESS"

        while not done and steps < max_steps:
            env.visited.add(env.current_pos)
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1

            frames.append((env.get_visualization().copy(), result, episode+1, steps))
            if steps == max_steps and not done:
                result = "MAX STEPS REACHED"


            if done:
                if reward == 20:
                    result = "SUCCESS"
                elif reward == -15:
                    result = "FAILURE (hit wall)"
            
                else:
                    result = "DONE"
                frames[-1] = (env.get_visualization().copy(), result, episode+1, steps)  # update last frame
                break

        episodes.append(frames)

    def update(frame):
        ax.clear()
        episode_idx, step_idx = frame
        grid, result, current_episode, current_step = episodes[episode_idx][step_idx]
        ax.imshow(grid, cmap=env.cmap, norm=env.norm)
        ax.set_xticks([])
        ax.set_yticks([])
        actual_episode = list(saved_agents.keys())[episode_idx]
        ax.set_title(f"Episode {actual_episode}, Step {current_step}\n{result}")
        return ax

    frames = [(ep_idx, step_idx)
              for ep_idx in range(len(episodes))
              for step_idx in range(len(episodes[ep_idx]))]

    ani = FuncAnimation(fig, update, frames=frames, interval=300, repeat=False)
    plt.show()
    agent.epsilon = original_epsilon  # Restore exploration
    return ani

def train_and_visualize(num_training_episodes=2000, max_steps=100):
    env = ChallengingMazeEnv(size=12)
    env.max_steps = max_steps
    agent = ImperfectQLearningAgent(env)

    print(f"Training agent for {num_training_episodes} episodes...")
    checkpoints = [1, num_training_episodes//4, num_training_episodes//2, 
                  num_training_episodes*3//4, num_training_episodes]
    saved_agents = {}

    for episode in range(1, num_training_episodes + 1):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

        if episode in checkpoints:
            saved_agents[episode] = ImperfectQLearningAgent(env)
            saved_agents[episode].q_table = np.copy(agent.q_table)
            saved_agents[episode].epsilon = agent.epsilon
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon: {agent.epsilon:.4f}")

    print("\nShowing agent performance at training stages:")
    for ep_num in checkpoints:
        print(f"\n--- Episode {ep_num} ---")
        animate_episodes(env, saved_agents[ep_num], ep_num, max_steps)
    
    print("\nFinal Learned Policy:")
    print_learned_policy(env, agent)

def print_learned_policy(env, agent):
    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    policy_grid = []

    for i in range(env.size):
        row = []
        for j in range(env.size):
            pos = (i, j)
            state = i * env.size + j

            if env.maze[i, j] == 1:
                row.append('■')  # Wall
            elif pos == env.goal_pos:
                row.append('G')  # Goal
            elif pos == env.start_pos:
                row.append('S')  # Start
            else:
                best_action = np.argmax(agent.q_table[state])
                row.append(arrow_map[best_action])
        policy_grid.append(row)

    print("\nLearned Policy:")
    for row in policy_grid:
        print(" ".join(row))



def animate_episodes(env, agent, episode_num, max_steps=100):
    fig, ax = plt.subplots(figsize=(8, 8))
    frames = []

    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Turn off exploration for evaluation

    state = env.reset()
    done = False
    steps = 0
    result = "IN PROGRESS"

    while not done and steps < max_steps:
        env.visited.add(env.current_pos)
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1

        frames.append((env.get_visualization().copy(), result, steps))

        if done:
            if reward == 20:
                result = "SUCCESS"
            elif reward == -15:
                result = "FAILURE (hit wall)"
            elif reward == -10:  # Max steps penalty case
                result = "MAX STEPS REACHED"
            else:
                result = "DONE"
            frames[-1] = (env.get_visualization().copy(), result, steps)
            break

    def update(frame):
        ax.clear()
        grid, result, current_step = frames[frame]
        ax.imshow(grid, cmap=env.cmap, norm=env.norm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Episode {episode_num}, Step {current_step}\n{result}")
        return ax

    ani = FuncAnimation(fig, update, frames=len(frames), interval=300, repeat=False)
    plt.show()
    agent.epsilon = original_epsilon
    return ani

if __name__ == "__main__":
    train_and_visualize(
        num_training_episodes=2000,  # Total training episodes
        max_steps=24                # Max steps per evaluation episode
    )
