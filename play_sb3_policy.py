import time
from stable_baselines3 import PPO
from asteroid_env import AsteroidAvoidEnv

def main():
    env = AsteroidAvoidEnv(render_mode="human")
    model = PPO.load("ppo_asteroid_avoid", env=env)

    for ep in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.15)

        # Use steps_remaining (and lives) to decide outcome
        steps_remaining = info.get("steps_remaining", None)
        lives = info.get("lives", None)

        if steps_remaining == 0 and lives is not None and lives > 0:
            outcome = "WON! 🎉"
        else:
            outcome = "LOST 💥"

        print(
            f"Episode {ep+1}: total_reward = {total_reward:.1f} | "
            f"steps_remaining = {steps_remaining}, lives = {lives} → {outcome}"
        )

        # pause between episodes
        time.sleep(1.5)

    env.close()

if __name__ == "__main__":
    main()
