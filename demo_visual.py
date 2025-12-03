"""
Visual demo of AsteroidAvoidEnv
Opens a PyGame window showing the game in action.
Press Ctrl+C to stop.
"""

import time
from asteroid_env import AsteroidAvoidEnv


def run_visual_demo():
    """Run the environment with visual rendering."""
    print("=" * 60)
    print("ASTEROID AVOID - VISUAL DEMO")
    print("=" * 60)
    print("\nStarting visual demo with render_mode='human'...")
    print("You should see a PyGame window open.")
    print("\nControls: The agent takes random actions automatically")
    print("Press Ctrl+C in the terminal to stop.\n")
    
    # Create environment with human rendering
    env = AsteroidAvoidEnv(render_mode="human")
    
    episode = 1
    
    try:
        while True:
            print(f"\n--- Episode {episode} ---")
            obs, info = env.reset()
            
            total_reward = 0
            steps = 0
            
            while True:
                # Take random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                # Add a small delay so you can see what's happening
                time.sleep(0.1)  # 10 FPS
                
                if terminated or truncated:
                    break
            
            print(f"Episode {episode} finished!")
            print(f"  Steps survived: {steps}")
            print(f"  Total reward: {total_reward:.1f}")
            print(f"  Outcome: {'WON! 🎉' if reward == 100 else 'LOST (collision) 💥'}")
            
            episode += 1
            time.sleep(1)  # Pause between episodes
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    finally:
        env.close()
        print("Window closed. Goodbye!")


if __name__ == "__main__":
    run_visual_demo()
