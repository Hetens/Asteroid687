import numpy as np
import torch
import os
# Using asteroid_env_1 as used in td_scratch.py
from asteroid_env_1 import AsteroidAvoidEnv
from td_scratch import ActorCriticAgent

def evaluate_td_policy(model_path, episodes=100):
    print(f"Loading environment...")
    env = AsteroidAvoidEnv(render_mode=None)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    print(f"Initializing agent...")
    # Note: Hidden dimensions must match what was trained. 
    # td_scratch.py uses defaults or hardcoded values in ActorCriticAgent.__init__ which calls ActorNetwork/CriticNetwork
    # ActorCriticAgent init: self.actor = ActorNetwork(..., hidden_dim_l1=128, hidden_dim_l2=64)
    # So we just instantiate ActorCriticAgent with state/action dims.
    agent = ActorCriticAgent(state_dim, action_dim)
    
    # Load model
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Error: Model not found at {model_path}")
        return

    wins = 0
    total_steps = 0
    
    print(f"Starting evaluation over {episodes} episodes...")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            # Deterministic action selection (mean of Gaussian)
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean, std = agent.actor(state_tensor)
            
            # Use mean for deterministic evaluation
            action = mean.detach().numpy().flatten()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
            
        # Check if won (survived all steps)
        if info["steps_remaining"] == 0 and info["lives"] > 0:
            wins += 1
            
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} completed. Current Win Rate: {wins/(ep+1):.2%}")
            
    env.close()
    
    win_rate = wins / episodes
    avg_steps = total_steps / episodes
    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"Model: {model_path}")
    print(f"Win Rate: {win_rate:.2%} ({wins}/{episodes})")
    print(f"Average Steps: {avg_steps:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    # Path to the saved model from td_scratch.py
    model_path = "logs/td_actor_policy.pth"
    evaluate_td_policy(model_path, episodes=100)
