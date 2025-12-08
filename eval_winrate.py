from asteroid_env import AsteroidAvoidEnv
from stable_baselines3 import TD3, PPO, SAC

POLICY = "PPO"
def evaluate_policy(model_path, episodes=100):
    env = AsteroidAvoidEnv(render_mode=None)
    if POLICY == "PPO":
        model = PPO.load(model_path, env=env)
    elif POLICY == "TD3":
        model = TD3.load(model_path, env=env)
    elif POLICY == "SAC":
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Invalid policy: {POLICY}")

    wins = 0

    for _ in range(episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info["steps_remaining"] == 0 and info["lives"] > 0:
            wins += 1

    env.close()
    print(f"Win rate over {episodes} episodes: {wins / episodes:.2f}")

if __name__ == "__main__":
    evaluate_policy(f"{POLICY}_asteroid_avoid", episodes=100)
