import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from asteroid_en import AsteroidAvoidEnv
import os


#Actor Network

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim_l1=64, hidden_dim_l2=64):
        super().__init__()
        #input layer
        self.fc1 = nn.Linear(state_dim, hidden_dim_l1)
        #hidden layer
        self.fc2 = nn.Linear(hidden_dim_l1, hidden_dim_l2)
        #output layer
        self.mean = nn.Linear(hidden_dim_l2, action_dim)
        #log standard deviation for each action range [-1, 1]
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        
    def forward(self, state):
        #forward pass if not work try tanh or leaky relu
        x = torch.relu(self.fc1(state))
        #hidden layer
        x = torch.relu(self.fc2(x))
        #output layer
        mean = torch.tanh(self.mean(x))
        #standard deviation for each action range [-1, 1] expanded to be the same shape as mean
        std = torch.exp(self.log_std).expand_as(mean)
        #clamp standard deviation between 0.1 and 1
        std = torch.clamp(std, min=0.1, max=1.0)
        return mean, std
    
    def get_action(self, state):
        #calculate mean and std from the forward pass
        mean, std = self.forward(state)
        #create a normal distribution with the mean and std
        dist = torch.distributions.Normal(mean, std)
        #sample an action from the distribution
        action = dist.sample()
        #calculate the log probability of the action
        log_prob = dist.log_prob(action).sum()
        return action.clamp(-1, 1), log_prob

#Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim_l1=64, hidden_dim_l2=64):
        super().__init__()
        #input layer
        self.fc1 = nn.Linear(state_dim, hidden_dim_l1)
        #hidden layer
        self.fc2 = nn.Linear(hidden_dim_l1, hidden_dim_l2)
        #output layer
        self.value = nn.Linear(hidden_dim_l2, 1)
        
    def forward(self, state):
        #forward pass
        x = torch.relu(self.fc1(state))
        #hidden layer
        x = torch.relu(self.fc2(x))
        #value(x) value of the state the critic estimates
        return self.value(x)


class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99):
        #discount
        self.gamma = gamma
        #actor network initialization
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim_l1=256, hidden_dim_l2=128)
        #critic network initialization
        self.critic = CriticNetwork(state_dim, hidden_dim_l1=256, hidden_dim_l2=128)
        #actor optimizer initialization
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        #critic optimizer initialization
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
    def select_action(self, state):
        """Select an action based on the current policy from the actor network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.get_action(state_tensor)
            #convert the action to a numpy array for the environment
            action_np = action.numpy().flatten()
        return action_np, state_tensor
    
    def update_batch(self, states, actions, rewards, next_states, dones):
        # Convert lists to tensors of batch dimension

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        next_states_t = torch.FloatTensor(np.array(next_states))
        rewards_t = torch.FloatTensor(rewards)
        dones_t = torch.FloatTensor(dones)
        
        #CRITIC UPDATE-------------------------
        #current estimates of critic's values of states
        values = self.critic(states_t).squeeze()
        next_values = self.critic(next_states_t).squeeze().detach()
        
        #TD target = R_t + gamma * V(s_{t+1}) * (whether or not episode terminated if terminated then 0 else 1)
        td_targets = rewards_t + self.gamma * next_values * (1 - dones_t)
        #TD error = TD target - V(s_t) what the critic estimates the value of the state to be
        td_errors = td_targets - values
        
        #LOSS FUNCTIONS ------

        #mse loss
        # critic_loss = td_errors.pow(2).mean()
        #huber loss 
        critic_loss = nn.functional.smooth_l1_loss(values, td_targets)

        #gradient update
        self.critic_optimizer.zero_grad()
        #backpropagate the loss
        critic_loss.backward()
        #clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        #update the critic
        self.critic_optimizer.step()
        
        #ACTOR UPDATE-------------------------
        # Recompute log probs with gradients enabled
        means, stds = self.actor(states_t)
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(actions_t).sum(dim=1)
        #             gradient of (log(pi(a|s)) * td_target - V(s)) w.r.t.the parameters of the actor
        actor_loss = -(log_probs * td_errors.detach()).mean()
        
        self.actor_optimizer.zero_grad()
        #backpropagate the loss
        actor_loss.backward()
        #clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        #update the actor
        self.actor_optimizer.step()
        
        # Return aggregated metrics
        return {
            'td_error': td_errors.mean().item(),
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'value_estimate': values.mean().item(),
            #exploration parameter is decided by the standard deviation of the action distribution
            'std': torch.exp(self.actor.log_std).mean().item()
        }
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


def train():
    env = AsteroidAvoidEnv(render_mode=None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99
    )
    
    num_episodes = 500
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    td_errors = []
    value_estimates = []
    policy_stds = []
    win_rate_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        # Collect entire episode in memory
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        total_reward = 0
        steps = 0
        done = False
        
        # Rollout episode
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1.0 if done else 0.0)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        # Single batched update for entire episode
        metrics = agent.update_batch(states, actions, rewards, next_states, dones)
        
        # Log metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        td_errors.append(metrics['td_error'])
        value_estimates.append(metrics['value_estimate'])
        policy_stds.append(metrics['std'])
        
        # Win rate
        if episode >= 99:
            wins = sum(1 for length in episode_lengths[-100:] if length >= 1000)
            win_rate_history.append(wins)
        else:
            win_rate_history.append(0)
        
        # Progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_td = np.mean(td_errors[-50:])
            std = policy_stds[-1]
            wins = win_rate_history[-1] if win_rate_history else 0
            
            print(f"Ep {episode+1:4d} | "
                  f"R: {avg_reward:4.4f} | "
                  f"Len: {avg_length:4.4f} | "
                  f"TD: {avg_td:4.4f} | "
                  f"σ: {std:4.4f} | "
                  f"Win: {wins:2d}/100")
    
    env.close()
    
    return agent, {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'td_errors': td_errors,
        'value_estimates': value_estimates,
        'policy_stds': policy_stds,
        'win_rate': win_rate_history
    }


def plot_training(metrics, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Episode Rewards
    ax = axes[0, 0]
    ax.plot(metrics['rewards'], alpha=0.2, color='blue')
    ax.plot(smooth(metrics['rewards']), color='blue', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode Lengths
    ax = axes[0, 1]
    ax.plot(metrics['lengths'], alpha=0.2, color='green')
    ax.plot(smooth(metrics['lengths']), color='green', linewidth=2, label='Smoothed')
    ax.axhline(y=1000, color='red', linestyle='--', label='Goal', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps Survived')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TD Error
    ax = axes[1, 0]
    ax.plot(metrics['td_errors'], alpha=0.3, color='purple')
    ax.plot(smooth(metrics['td_errors']), color='purple', linewidth=2, label='Smoothed')
    #WHERE THE FINAL ERROR SHOULD BE
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg TD Error')
    ax.set_title('TD Error (+ = Underestimate, - = Overestimate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Value Estimates
    ax = axes[1, 1]
    ax.plot(metrics['value_estimates'], alpha=0.3, color='orange')
    ax.plot(smooth(metrics['value_estimates']), color='orange', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg V(s)')
    ax.set_title('Value Function Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Exploration
    ax = axes[2, 0]
    ax.plot(metrics['policy_stds'], alpha=0.3, color='red')
    ax.plot(smooth(metrics['policy_stds']), color='red', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Policy Std')
    ax.set_title('Exploration (σ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Win Rate
    ax = axes[2, 1]
    ax.plot(metrics['win_rate'], color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Wins / 100')
    ax.set_title('Win Rate (Last 100 Episodes)')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlots saved to {save_path}")
    plt.close()


def evaluate(agent, num_episodes=100, render=False):
    """Evaluate trained agent"""
    env = AsteroidAvoidEnv(render_mode='human' if render else None)
    
    wins = 0
    lengths = []
    rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated
            total_reward += reward
            steps += 1
        
        lengths.append(steps)
        rewards.append(total_reward)
        if steps >= 1000:
            wins += 1
    
    env.close()
    
    print(f"Win Rate:    {wins}/{num_episodes} = {100*wins/num_episodes:.1f}%")
    print(f"Avg Length:  {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Avg Reward:  {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    
    return wins, lengths, rewards


if __name__ == "__main__":
    import time

    start = time.time()
    agent, metrics = train()
    elapsed = time.time() - start
    
    print(f"\n Training completed in {elapsed/60:.1f} minutes")
    
    # Save
    os.makedirs("logs", exist_ok=True)
    agent.save("logs/fast_td_policy.pth")
    plot_training(metrics, "logs/fast_td_training.png")
    
    # Evaluate
    print("\nEvaluating agent...")
    evaluate(agent, num_episodes=100, render=False)