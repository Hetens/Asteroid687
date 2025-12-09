"""
Temporal Difference Learning Agent for Asteroid Environment
Using Actor-Critic with Neural Network Function Approximation

TD Error: δ = r + γ * V(s') - V(s)

================================================================================
KEY CONCEPTS EXPLAINED:
================================================================================

1. EXPLORATION VS EXPLOITATION:
   - Exploration is controlled by the standard deviation (std) of the Gaussian policy
   - `log_std` is a LEARNABLE parameter that starts at 0 (std = 1.0)
   - High std → more random actions → EXPLORATION
   - Low std → actions closer to mean → EXPLOITATION
   - The agent learns to reduce std over time as it becomes more confident

2. HOW THE CRITIC UNDERSTANDS LOSS:
   - The Critic predicts V(s) = "expected future reward from state s"
   - TD Target = r + γ * V(s') = "what V(s) should actually be"
   - TD Error = TD Target - V(s) = "how wrong was my prediction?"
   - Loss = TD_Error² → gradient descent pushes V(s) toward the correct value
   - Intuition: If we underestimate (δ > 0), push V(s) up. If overestimate (δ < 0), push down.

3. HOW THE CRITIC TEACHES THE ACTOR:
   - The TD error δ acts as a "feedback signal" to the Actor
   - δ > 0 (positive): "This action was BETTER than expected" → INCREASE its probability
   - δ < 0 (negative): "This action was WORSE than expected" → DECREASE its probability  
   - δ ≈ 0: "This action was about as expected" → little change
   - Actor loss = -log π(a|s) * δ
   - This is policy gradient with the TD error as an "advantage" estimate

================================================================================
"""
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from asteroid_env_1 import AsteroidAvoidEnv
import os


# ============== Neural Network Models for Policy Estimation from Observations ==============

class ActorNetwork(nn.Module):
    """
    Policy Network (Actor): Maps state -> action distribution
    Outputs mean and std for a Gaussian policy
    
    The Gaussian policy allows continuous actions:
    - mean: the "best guess" action for this state
    - std: how much to explore around that guess
    """
    def __init__(self, state_dim, action_dim, hidden_dim_l1=64, hidden_dim_l2=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim_l1)
        self.fc2 = nn.Linear(hidden_dim_l1, hidden_dim_l2)
        self.mean = nn.Linear(hidden_dim_l2, action_dim)
        
        # log_std is LEARNABLE - this controls exploration!
        # Starts at 0 → std = exp(0) = 1.0 (high exploration)
        # As training progresses, typically decreases → less exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))  # bound to [-1, 1]
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def get_action(self, state):
        """Sample action from policy"""
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        # log_prob = log π(a|s) - used for policy gradient
        log_prob = dist.log_prob(action).sum()
        return action.clamp(-1, 1), log_prob


class CriticNetwork(nn.Module):
    """
    Value Network (Critic): Maps state -> V(s)
    
    V(s) = Expected sum of future discounted rewards starting from state s
    The Critic's job is to estimate "how good is this state?"
    """
    def __init__(self, state_dim, hidden_dim_l1=64, hidden_dim_l2=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim_l1)
        self.fc2 = nn.Linear(hidden_dim_l1, hidden_dim_l2)
        self.value = nn.Linear(hidden_dim_l2, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.value(x)


# ============== Actor-Critic Agent ==============

class ActorCriticAgent:
    """
    TD Actor-Critic Agent
    
    The Critic learns V(s) using TD(0):
        V(s) ← V(s) + α * [r + γ*V(s') - V(s)]
        
    The Actor learns π(a|s) using policy gradient with TD error as advantage:
        θ ← θ + α * δ * ∇log π(a|s)
    """
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim_l1=128, hidden_dim_l2=64)
        self.critic = CriticNetwork(state_dim, hidden_dim_l1=128, hidden_dim_l2=64)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
    def select_action(self, state):
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.actor.get_action(state_tensor)
        return action.detach().numpy().flatten(), log_prob
    
    def update(self, state, action, reward, next_state, done, log_prob):
        """
        Perform one TD update step - THIS IS WHERE LEARNING HAPPENS!
        
        The Critic learns by minimizing TD error (prediction error).
        The Actor learns by using TD error as feedback on action quality.
        """
        # Convert to tensors
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        reward_t = torch.FloatTensor([reward])
        done_t = torch.FloatTensor([1.0 if done else 0.0])
        
        # ===== CRITIC UPDATE (TD Learning) =====
        # Step 1: Get current value estimate V(s)
        value = self.critic(state_t)
        
        # Step 2: Get next state value V(s') - detached so we don't backprop through it
        next_value = self.critic(next_state_t).detach()
        
        # Step 3: Compute TD Target = r + γ * V(s') * (1 - done)
        # This is what V(s) SHOULD be based on actual reward + future estimate
        td_target = reward_t + self.gamma * next_value * (1 - done_t)
        
        # Step 4: Compute TD Error δ = TD_target - V(s)
        # δ > 0 means we underestimated, δ < 0 means we overestimated
        td_error = td_target - value
        
        # Step 5: Critic loss = δ² (minimize squared prediction error)
        critic_loss = td_error.pow(2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== ACTOR UPDATE (Policy Gradient with TD error as advantage) =====
        # The TD error tells the Actor whether the action was good or bad:
        # δ > 0: Action led to better-than-expected outcome → increase probability
        # δ < 0: Action led to worse-than-expected outcome → decrease probability

        # Actor loss = -log π(a|s) * δ

        # The negative sign is because we want to MAXIMIZE expected return (gradient ascent)
        actor_loss = -log_prob * td_error.detach()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return td_error.item(), critic_loss.item(), actor_loss.item()
    
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
    
    # Create agent
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99
    )
    
    # Training parameters
    num_episodes = 3000
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    all_actor_losses = []
    all_critic_losses = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
        ep_actor_losses = []
        ep_critic_losses = []
        
        while not done:
            # Select action from current policy
            action, log_prob = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
            # Update agent with TD learning
            td_error, critic_loss, actor_loss = agent.update(
                state, action, reward, next_state, done, log_prob
            )
            
            # Track losses
            ep_actor_losses.append(actor_loss)
            ep_critic_losses.append(critic_loss)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        # Log episode
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        all_actor_losses.append(np.mean(ep_actor_losses))
        all_critic_losses.append(np.mean(ep_critic_losses))
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f}")
    
    env.close()
    return agent, episode_rewards, episode_lengths, all_actor_losses, all_critic_losses


def plot_training(rewards, lengths, actor_losses, critic_losses, save_path):
    """Create 2x2 plot of training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Smoothing function
    def smooth(data, window=20):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(smooth(rewards), color='blue', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode Lengths
    ax2 = axes[0, 1]
    ax2.plot(lengths, alpha=0.3, color='green', label='Raw')
    ax2.plot(smooth(lengths), color='green', linewidth=2, label='Smoothed')
    ax2.axhline(y=1000, color='red', linestyle='--', label='Goal (1000 steps)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps Survived')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Actor Loss
    ax3 = axes[1, 0]
    ax3.plot(actor_losses, alpha=0.3, color='orange', label='Raw')
    ax3.plot(smooth(actor_losses), color='orange', linewidth=2, label='Smoothed')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Actor Loss (Policy Gradient)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Critic Loss
    ax4 = axes[1, 1]
    ax4.plot(critic_losses, alpha=0.3, color='red', label='Raw')
    ax4.plot(smooth(critic_losses), color='red', linewidth=2, label='Smoothed')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.set_title('Critic Loss (TD Error²)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training plots saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Starting TD Actor-Critic Training...")
    print("=" * 50)
    
    agent, rewards, lengths, actor_losses, critic_losses = train()
    
    # Save policy
    os.makedirs("logs", exist_ok=True)
    agent.save("logs/td_actor_policy.pth")
    
    # Plot training metrics
    plot_training(rewards, lengths, actor_losses, critic_losses, "logs/td_scratch_training.png")
    
    print("=" * 50)
    print("Training complete!")
    print(f"Final avg reward (last 10): {np.mean(rewards[-10:]):.2f}")
    print(f"Final avg length (last 10): {np.mean(lengths[-10:]):.2f}")
