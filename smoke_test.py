from asteroid_env import AsteroidAvoidEnv

def run_smoke_test():
    print("Running headless smoke test...")
    env = AsteroidAvoidEnv(render_mode=None)

    for ep in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep+1}: total_reward = {total_reward}")

    env.close()
    print("Smoke test completed.")

if __name__ == "__main__":
    run_smoke_test()
