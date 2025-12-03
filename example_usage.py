"""
Example usage of the Asteroids Gymnasium Environment

This script demonstrates:
1. How to create the environment
2. Random action sampling
3. Episode management
4. Rendering options
"""

import gymnasium as gym
import asteroids_env
import numpy as np


def random_agent_demo(render=True, num_episodes=3):
    """
    Run a random agent for demonstration purposes.
    
    Args:
        render: Whether to render the environment
        num_episodes: Number of episodes to run
    """
    # Create environment
    render_mode = "human" if render else None
    env = gym.make("asteroids_env/Asteroids-v0", render_mode=render_mode)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        step_count = 0
        
        print(f"Episode {episode + 1} started")
        print(f"Initial info: {info}")
        
        while True:
            # Sample random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Check if episode ended
            if terminated or truncated:
                print(f"Episode {episode + 1} finished:")
                print(f"  Steps: {step_count}")
                print(f"  Total reward: {episode_reward:.2f}")
                print(f"  Terminated: {terminated} (collision)" if terminated else "  Truncated: True (time limit)")
                print(f"  Final info: {info}")
                print()
                break
    
    env.close()


def smart_agent_demo(render=True, num_episodes=3):
    """
    A slightly smarter agent that tries to avoid asteroids and shoot them.
    
    Args:
        render: Whether to render the environment
        num_episodes: Number of episodes to run
    """
    render_mode = "human" if render else None
    env = gym.make("asteroids_env/Asteroids-v0", render_mode=render_mode)
    
    print("Running smarter agent (avoids close asteroids)")
    print()
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=100 + episode)
        episode_reward = 0
        step_count = 0
        
        print(f"Episode {episode + 1} started")
        
        while True:
            # Extract observation components
            player_state = obs["player"]
            nearest_asteroids = obs["nearest_asteroids"]
            shoot_ready = obs["shoot_ready"]
            
            # Simple strategy:
            # 1. If asteroid is very close, move away
            # 2. Otherwise, rotate toward nearest asteroid and shoot
            
            # Check if any asteroid is dangerously close
            min_distance = float('inf')
            closest_ast_dir = None
            
            for i in range(5):
                rel_pos = nearest_asteroids[i, :2]
                if np.any(rel_pos != 0):  # Valid asteroid
                    dist = np.linalg.norm(rel_pos)
                    if dist < min_distance:
                        min_distance = dist
                        closest_ast_dir = rel_pos
            
            # Default action: no rotation, no thrust, no shoot
            rotation_action = 1  # none
            thrust_action = 1    # none
            shoot_action = 0     # no
            
            if closest_ast_dir is not None:
                # Calculate angle to closest asteroid
                angle_to_ast = np.degrees(np.arctan2(closest_ast_dir[0], closest_ast_dir[1]))
                current_rotation = player_state[4]
                
                # Normalize angle difference
                angle_diff = (angle_to_ast - current_rotation + 180) % 360 - 180
                
                if min_distance < 150:
                    # Danger! Move away
                    if abs(angle_diff) < 30:
                        # Facing asteroid, rotate away
                        rotation_action = 0 if angle_diff > 0 else 2
                        thrust_action = 0  # backward
                    else:
                        # Try to face away and thrust
                        if abs(angle_diff) > 150:
                            thrust_action = 2  # forward (away from asteroid)
                        else:
                            rotation_action = 2 if angle_diff < 0 else 0
                else:
                    # Safe, try to shoot asteroids
                    if abs(angle_diff) < 20:
                        # Aligned, shoot!
                        if shoot_ready:
                            shoot_action = 1
                    else:
                        # Rotate toward asteroid
                        rotation_action = 2 if angle_diff > 0 else 0
            
            action = np.array([rotation_action, thrust_action, shoot_action])
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"Episode {episode + 1} finished:")
                print(f"  Steps: {step_count}")
                print(f"  Total reward: {episode_reward:.2f}")
                print(f"  Asteroids destroyed: {info.get('num_asteroids', 'N/A')}")
                print()
                break
        
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Asteroids Environment Demo")
    parser.add_argument("--mode", choices=["random", "smart"], default="random",
                        help="Agent type: random or smart")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    
    args = parser.parse_args()
    
    render = not args.no_render
    
    if args.mode == "random":
        random_agent_demo(render=render, num_episodes=args.episodes)
    else:
        smart_agent_demo(render=render, num_episodes=args.episodes)
