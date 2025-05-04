# Simple-RL-agent-in-a-maze

A Q-learning agent trained to navigate a complex maze environment with dynamic penalties and visualizations. Developed with assistance from ChatGPT.
This code was developed for a presentation for Introduction to machine learning course 'BME4462' in Ankara University.

## Description
This project implements a Q-learning algorithm to train an agent to navigate a challenging 12x12 maze. The environment includes walls, traps, and a goal position, with the agent receiving rewards/penalties based on its actions. The solution includes animated visualizations of the agent's learning progress and a final learned policy.

## Key Features
- **Complex Maze Design**: Predefined walls and obstacles creating a non-trivial navigation challenge.
- **Q-Learning Agent**: Implements exploration-exploitation tradeoff with epsilon decay.
- **Animated Training Progress**: Visualizes the agent's movement, visited states, and best path.
- **Dynamic Penalties**: 
  - `-15` for hitting walls
  - `-10` for exceeding step limit
  - `+20` for reaching goal
- **Policy Visualization**: Displays learned directions (↑↓←→) for each position.

## Installation
```bash
pip install numpy matplotlib
```