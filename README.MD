# Asteroid Avoidance RL Project

This project trains an agent to control a ship and avoid falling asteroids in a custom Gymnasium environment. The agent is trained using three deep RL algorithms: PPO, SAC, and TD3, all implemented with Stable-Baselines3.

## Environment Description

Goal: Survive 1000 steps without losing all 3 lives
Action: Continuous acceleration in [-1, 1]
Observation: Ship + asteroid positions/velocities + progress indicators
Reward: Small reward for staying alive, bonuses for staying centered/avoiding asteroids, penalties for collisions/walls, large bonus for surviving the episode

## Setup

### Prerequisites

#### Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train agents using one of the three algorithms:

#### PPO (Proximal Policy Optimization)

```bash
python train_ppo_sb3.py
```

#### SAC (Soft Actor-Critic)

```bash
python train_sac_sb3.py
```

#### TD3 (Twin Delayed DDPG)

```bash
python train_td3_sb3.py
```

**Note**: Each script will train the model, log metrics, and save the final weights and training plots in the project folder.

### Evaluation

Evaluate a trained model's win rate:

```bash
python eval_winrate.py
```

Edit the `POLICY` variable in `eval_winrate.py` to switch between "PPO", "SAC", or "TD3". The script runs 100 episodes and reports the win rate.

### Visualization

Play a trained policy with visual rendering:

```bash
python play_sb3_policy.py
```

Edit the `POLICY` variable in `play_sb3_policy.py` to select which model to visualize. The script will:

- Load the trained model
- Run 5 episodes with visual rendering
- Display statistics (wins, losses, rewards)
