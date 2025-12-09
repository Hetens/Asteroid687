"""
Temporal Difference Learning Agent for Asteroid Environment
Using Actor-Critic with Neural Network Function Approximation

IMPROVED VERSION with:
- Batch updates instead of online TD(0)
- Generalized Advantage Estimation (GAE)
- Entropy regularization for exploration
- Gradient clipping for stability
- Bounded exploration (log_std) ---> dont know if necessary

================================================================================
"""
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from asteroid_env_1 import AsteroidAvoidEnv
import os


# ============== Neural Network Models ==============

class ActorNetwork(nn.Module):
    """
    Policy Network (Actor): Maps state -> action distribution
    Outputs mean and std for a Gaussian policy
    """
    def __init__(self, state_dim, action_dim, hidden_dim_l1=128, hidden_dim_l2=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim_l1)
        self.fc2 = nn.Linear(hidden_dim_l1, hidden_dim_l2)
        self.mean = nn.Linear(hidden_dim_l2, action_dim)
        
        # FIXED: Bounded log_std to prevent exploration collapse or explosion
        # Range: exp(-2) = 0.135 to exp(0.5) = 1.65
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = -2.0
        self.log_std_max = 0.5
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))  # bound to [-1, 1]
        
        # Clamp log_std to prevent collapse
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        mean, std = self.forward(state)
        if deterministic:
            return mean.clamp(-1, 1), None, None
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action.clamp(-1, 1), log_prob, entropy
    
    def evaluate_actions(self, states, actions):
        """Evaluate log_prob and entropy for given state-action pairs"""
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """
    Value Network (Critic): Maps state -> V(s)
    """
    def __init__(self, state_dim, hidden_dim_l1=128, hidden_dim_l2=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim_l1)
        self.fc2 = nn.Linear(hidden_dim_l1, hidden_dim_l2)
        self.value = nn.Linear(hidden_dim_l2, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.value(x).squeeze(-1)


# ============== Rollout Buffer for Batch Updates ==============

class RolloutBuffer:
    """Store trajectories for batch updates"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def __len__(self):
        return len(self.states)


# ============== Actor-Critic Agent with GAE ==============

class ActorCriticAgent:
    """
    Improved TD Actor-Critic Agent with:
    - Generalized Advantage Estimation (GAE)
    - Entropy regularization
    - Gradient clipping
    - Batch updates
    """
    def __init__(self, state_dim, action_dim, 
                 lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, gae_lambda=0.95,
                 entropy_coef=0.01, max_grad_norm=0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
    def select_action(self, state):
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor)
        action, log_prob, entropy = self.actor.get_action(state_tensor)
        return (action.numpy().flatten(), 
                log_prob.item(), 
                value.item())
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store transition in buffer"""
        self.buffer.add(state, action, reward, done, log_prob, value)
    
    def compute_gae(self, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        GAE(γ,λ) = Σ (γλ)^t * δ_t
        where δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        
        This provides a good bias-variance tradeoff for advantage estimation.
        """
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [next_value]
        
        advantages = []
        gae = 0
        
        # Compute GAE in reverse order
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(self.buffer.values)
        
        # Normalize advantages (crucial for stability!)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, next_state):
        """
        Perform batch update at end of episode/rollout
        """
        if len(self.buffer) == 0:
            return 0, 0, 0
        
        # Get next state value for GAE computation
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            next_value = self.critic(next_state_t).item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer.states))
        actions = torch.FloatTensor(np.array(self.buffer.actions))
        old_log_probs = torch.FloatTensor(self.buffer.log_probs)
        
        # ===== CRITIC UPDATE =====
        values = self.critic(states)
        critic_loss = nn.functional.mse_loss(values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # ===== ACTOR UPDATE =====
        log_probs, entropy = self.actor.evaluate_actions(states, actions)
        
        # Policy gradient loss with entropy regularization
        actor_loss = -(log_probs * advantages).mean()
        entropy_loss = -entropy.mean()  # negative because we want to maximize entropy
        
        total_actor_loss = actor_loss + self.entropy_coef * entropy_loss
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Clear buffer after update
        self.buffer.clear()
        
        return advantages.mean().item(), critic_loss.item(), actor_loss.item()
    
    def save(self, path):
        """Save policy to file"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Policy saved to {path}")
    
    def load(self, path):
        """Load policy from file"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Policy loaded from {path}")


# ============== Training Loop ==============

def train():
    # Create environment
    env = AsteroidAvoidEnv(render_mode=None)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]  # 24
    action_dim = env.action_space.shape[0]      # 1
    
    # Create agent with improved hyperparameters
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,      # GAE lambda for advantage estimation
        entropy_coef=0.01,    # Entropy bonus for exploration
        max_grad_norm=0.5     # Gradient clipping
    )
    
    # Training parameters
    num_episodes = 5000
    update_frequency = 1  # Update after every episode
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    all_advantages = []
    all_critic_losses = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Select action from current policy
            action, log_prob, value = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
            # Store transition
            agent.store_transition(state, action, reward, done, log_prob, value)
            
            total_reward += reward  # Track original reward
            steps += 1
            state = next_state
        
        # Update at end of episode
        if (episode + 1) % update_frequency == 0:
            avg_advantage, critic_loss, actor_loss = agent.update(next_state)
            all_advantages.append(avg_advantage)
            all_critic_losses.append(critic_loss)
        
        # Log episode
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            current_std = torch.exp(agent.actor.log_std).item()
            print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:6.2f} | Exploration σ: {current_std:.3f}")
    
    env.close()
    return agent, episode_rewards, episode_lengths, all_advantages, all_critic_losses


def plot_training(rewards, lengths, advantages, critic_losses, save_path):
    """Create 2x2 plot of training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Smoothing function
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(smooth(rewards), color='blue', linewidth=2, label='Smoothed (50)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode Lengths
    ax2 = axes[0, 1]
    ax2.plot(lengths, alpha=0.3, color='green', label='Raw')
    ax2.plot(smooth(lengths), color='green', linewidth=2, label='Smoothed (50)')
    ax2.axhline(y=1000, color='red', linestyle='--', label='Goal (1000 steps)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps Survived')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Average Advantage
    ax3 = axes[1, 0]
    if len(advantages) > 0:
        ax3.plot(advantages, alpha=0.3, color='orange', label='Raw')
        ax3.plot(smooth(advantages, window=20), color='orange', linewidth=2, label='Smoothed')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Update')
    ax3.set_ylabel('Advantage')
    ax3.set_title('Average Advantage (should be ~0 after normalization)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Critic Loss
    ax4 = axes[1, 1]
    if len(critic_losses) > 0:
        ax4.plot(critic_losses, alpha=0.3, color='red', label='Raw')
        ax4.plot(smooth(critic_losses, window=20), color='red', linewidth=2, label='Smoothed')
    ax4.set_xlabel('Update')
    ax4.set_ylabel('Loss')
    ax4.set_title('Critic Loss (TD Error²)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training plots saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Starting Improved TD Actor-Critic Training...")
    print("=" * 60)
    print("Key improvements:")
    print("  - GAE for lower variance advantage estimation")
    print("  - Entropy regularization to maintain exploration")
    print("  - Gradient clipping for stability")
    print("  - Batch updates after each episode")
    print("=" * 60)
    
    agent, rewards, lengths, advantages, critic_losses = train()
    
    # Save policy
    os.makedirs("logs", exist_ok=True)
    agent.save("logs/td_actor_policy_v2.pth")
    
    # Plot training metrics
    plot_training(rewards, lengths, advantages, critic_losses, "logs/td_scratch_training_v2.png")
    
    print("=" * 60)
    print("Training complete!")
    print(f"Final avg reward (last 100): {np.mean(rewards[-100:]):.2f}")
    print(f"Final avg length (last 100): {np.mean(lengths[-100:]):.2f}")
    print(f"Best episode length: {max(lengths)}")
