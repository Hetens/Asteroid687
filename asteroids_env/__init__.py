from gymnasium.envs.registration import register

register(
    id="asteroids_env/Asteroids-v0",
    entry_point="asteroids_env.envs:AsteroidsEnv",
    max_episode_steps=3000,
)
