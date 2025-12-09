import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from asteroid_env import AsteroidAvoidEnv
from utils import SimpleLogger, save_hyperparams, plot_training_results, print_training_summary

LOG_DIR = "./ppo/logs/"
os.makedirs(LOG_DIR, exist_ok=True)

POLICY = "PPO"
N_ENVS = 16

HYPERPARAMS = {
    "total_timesteps": 25_000_000,
    "learning_rate": 3e-4,
    "n_steps": 1024,        
    "batch_size": 2048,     
    "n_epochs": 20,         
    "gamma": 0.99,    
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01, 
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
}


def main():
    print(f"Training {POLICY} with {N_ENVS} parallel envs")

    save_hyperparams(LOG_DIR, POLICY, N_ENVS, HYPERPARAMS)

    # Vectorized env
    env = make_vec_env(
        lambda: Monitor(AsteroidAvoidEnv()),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]), 
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        HYPERPARAMS["policy"],
        env,
        learning_rate=HYPERPARAMS["learning_rate"],
        n_steps=HYPERPARAMS["n_steps"],
        batch_size=HYPERPARAMS["batch_size"],
        n_epochs=HYPERPARAMS["n_epochs"],
        gamma=HYPERPARAMS["gamma"],
        gae_lambda=HYPERPARAMS["gae_lambda"],
        clip_range=HYPERPARAMS["clip_range"],
        ent_coef=HYPERPARAMS["ent_coef"],
        vf_coef=HYPERPARAMS["vf_coef"],
        max_grad_norm=HYPERPARAMS["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR, 
    )

    logger = SimpleLogger(
        os.path.join(LOG_DIR, "training_log.csv"),
        loss_key='train/value_loss'
    )

    print(f"\nStarting {POLICY} training...")

    model.learn(
        total_timesteps=HYPERPARAMS["total_timesteps"],
        callback=logger,
        progress_bar=True,
    )

    model.save(f"{POLICY}_asteroid_avoid")
    
    plot_training_results(logger, LOG_DIR, loss_label='Value Loss')
    print_training_summary(logger, POLICY)

    env.close()


if __name__ == "__main__":
    main()