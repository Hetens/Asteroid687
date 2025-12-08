import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from asteroid_env import AsteroidAvoidEnv
from utils import SimpleLogger, save_hyperparams, plot_training_results, print_training_summary

LOG_DIR = "./sac/logs/"
os.makedirs(LOG_DIR, exist_ok=True)

POLICY = "SAC"
N_ENVS = 8

HYPERPARAMS = {
    "total_timesteps": 5_000_000,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 300_000,
    "gamma": 0.97,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy": "MlpPolicy",
    "ent_coef": "auto",
    "target_entropy": "auto",
    "learning_starts": 5000,
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
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    model = SAC(
        HYPERPARAMS["policy"],
        env,
        learning_rate=HYPERPARAMS["learning_rate"],
        batch_size=HYPERPARAMS["batch_size"],
        buffer_size=HYPERPARAMS["buffer_size"],
        gamma=HYPERPARAMS["gamma"],
        tau=HYPERPARAMS["tau"],
        train_freq=HYPERPARAMS["train_freq"],
        gradient_steps=HYPERPARAMS["gradient_steps"],
        ent_coef=HYPERPARAMS["ent_coef"],
        target_entropy=HYPERPARAMS["target_entropy"],
        learning_starts=HYPERPARAMS["learning_starts"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR, 
    )

    logger = SimpleLogger(
        os.path.join(LOG_DIR, "training_log.csv"),
        loss_key='train/critic_loss'
    )

    print(f"\nStarting {POLICY} training...")

    model.learn(
        total_timesteps=HYPERPARAMS["total_timesteps"],
        callback=logger,
        progress_bar=True,
    )

    model.save(f"{POLICY}_asteroid_avoid")
    
    plot_training_results(logger, LOG_DIR, loss_label='Critic Loss')
    print_training_summary(logger, POLICY)

    env.close()


if __name__ == "__main__":
    main()