"""
Run trained TD Actor-Critic policy in visual mode
"""
import torch
import numpy as np
from asteroid_env import AsteroidAvoidEnv
from td_scratch import ActorCriticAgent


def run_policy(policy_path, num_episodes=5):
    """Load and run a trained policy with visualization"""
    
    # Create environment with rendering
    env = AsteroidAvoidEnv(render_mode="human")
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent and load policy
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=1e-4,  # not used for inference
        lr_critic=1e-3,
        gamma=0.99
    )
    agent.load(policy_path)
    
    # Set to evaluation mode (disables dropout, etc.)
    agent.actor.eval()
    agent.critic.eval()
    
    print(f"\nRunning {num_episodes} episodes with trained policy...")
    print("=" * 50)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Use mean action (no sampling) for deterministic behavior
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                mean, std = agent.actor(state_tensor)
                action = mean.numpy().flatten()  # use mean directly for evaluation
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
            total_reward += reward
            steps += 1
            state = next_state
        
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    env.close()
    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    run_policy("logs/td_actor_policy.pth", num_episodes=5)
