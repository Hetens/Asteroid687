from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from asteroid_env import AsteroidAvoidEnv

def make_env():
    env = AsteroidAvoidEnv(render_mode=None)
    env = Monitor(env)  # tracks episode rewards, lengths
    return env

def main():
    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)
    model.save("ppo_asteroid_avoid")
    env.close()

if __name__ == "__main__":
    main()
