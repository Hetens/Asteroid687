"""
Script to play/visualize a trained PPO policy on the Asteroid Avoid Environment.
"""

import os
import time
from stable_baselines3 import PPO
from asteroid_env import AsteroidAvoidEnv


def main():
    # Check for model file
    model_paths = [
        "ppo_asteroid_avoid.zip",
        "ppo_asteroid_avoid",
        "./models/ppo_asteroid_avoid_final.zip",
        "./models/ppo_asteroid_avoid_final",
        "./models/best_model.zip",
        "./models/best_model",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            model_path = path
            break
    
    if model_path is None:
        print("Error: No trained model found!")
        print("Please run train_sb3_baseline.py first to train a model.")
        print(f"Searched paths: {model_paths}")
        return
    
    print(f"Loading model from: {model_path}")
    
    env = AsteroidAvoidEnv(render_mode="human")
    model = PPO.load(model_path, env=env)
    
    print("\n" + "="*50)
    print("Playing trained policy...")
    print("="*50)
    print("Environment: Continuous Asteroid Avoid")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print("="*50 + "\n")

    num_episodes = 5
    total_wins = 0
    total_losses = 0
    all_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            time.sleep(0.033)  # ~30 FPS to match render_fps

        # Use steps_remaining (and lives) to decide outcome
        steps_remaining = info.get("steps_remaining", None)
        lives = info.get("lives", None)

        if steps_remaining == 0 and lives is not None and lives > 0:
            outcome = "WON! 🎉"
            total_wins += 1
        else:
            outcome = "LOST 💥"
            total_losses += 1
        
        all_rewards.append(total_reward)

        print(
            f"Episode {ep+1}/{num_episodes}: reward = {total_reward:.1f} | "
            f"steps = {step_count}, lives = {lives} → {outcome}"
        )

        # pause between episodes
        time.sleep(1.0)

    env.close()
    
    # Print summary
    print("\n" + "="*50)
    print("PLAYBACK SUMMARY")
    print("="*50)
    print(f"Episodes played: {num_episodes}")
    print(f"Wins: {total_wins} ({100*total_wins/num_episodes:.1f}%)")
    print(f"Losses: {total_losses} ({100*total_losses/num_episodes:.1f}%)")
    print(f"Average reward: {sum(all_rewards)/len(all_rewards):.1f}")
    print(f"Best reward: {max(all_rewards):.1f}")
    print(f"Worst reward: {min(all_rewards):.1f}")
    print("="*50)


if __name__ == "__main__":
    main()
