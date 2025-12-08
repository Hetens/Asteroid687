import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from asteroid_env import AsteroidAvoidEnv

LOG_DIR = "./td3/logs/"
os.makedirs(LOG_DIR, exist_ok=True)

POLICY = "TD3"
N_ENVS = 8

HYPERPARAMS = {
    "total_timesteps": 5_000_000,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 300_000,
    "gamma": 0.97,
    "tau": 0.005,
    "train_freq": 4,
    "gradient_steps": 4,
    "policy": "MlpPolicy",
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "learning_starts": 5000,
}


class SimpleLogger(BaseCallback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.returns = []
        self.losses = []
        self.last_ep_count = 0

        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['episode', 'return', 'length', 'loss'])

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.last_ep_count += 1
                ep_return = info['episode']['r']
                ep_length = info['episode']['l']

                loss = 0.0
                if 'train/critic_loss' in self.logger.name_to_value:
                    loss = self.logger.name_to_value['train/critic_loss']
                elif 'train/loss' in self.logger.name_to_value:
                    loss = self.logger.name_to_value['train/loss']

                self.returns.append(ep_return)
                self.losses.append(loss)

                with open(self.csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [self.last_ep_count, ep_return, ep_length, loss]
                    )

                if self.last_ep_count % 100 == 0:
                    recent_mean = np.mean(self.returns[-100:])
                    recent_max = max(self.returns[-100:])
                    # Calculate win rate
                    recent_wins = sum(1 for r in self.returns[-100:] if r > 1500)
                    win_rate = recent_wins / min(100, len(self.returns[-100:]))
                    print(
                        f"Ep {self.last_ep_count}: "
                        f"Mean={recent_mean:.1f}, Max={recent_max:.1f}, "
                        f"WinRate={win_rate:.2%}"
                    )
        return True


def save_hyperparams(path):
    with open(path, 'w') as f:
        f.write(f"Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{POLICY} Hyperparameters:\n")
        f.write(f"Number of parallel environments: {N_ENVS}\n")
        f.write("-" * 30 + "\n")
        for key, value in HYPERPARAMS.items():
            f.write(f"{key}: {value}\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {POLICY} with {N_ENVS} parallel envs on {device}...")

    save_hyperparams(os.path.join(LOG_DIR, "hyperparams.txt"))

    # Vectorized env
    env = make_vec_env(
        lambda: Monitor(AsteroidAvoidEnv()),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions)  # High exploration
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    model = TD3(
        HYPERPARAMS["policy"],
        env,
        learning_rate=HYPERPARAMS["learning_rate"],
        batch_size=HYPERPARAMS["batch_size"],
        buffer_size=HYPERPARAMS["buffer_size"],
        gamma=HYPERPARAMS["gamma"],
        tau=HYPERPARAMS["tau"],
        train_freq=HYPERPARAMS["train_freq"],
        gradient_steps=HYPERPARAMS["gradient_steps"],
        policy_delay=HYPERPARAMS["policy_delay"],
        target_policy_noise=HYPERPARAMS["target_policy_noise"],
        target_noise_clip=HYPERPARAMS["target_noise_clip"],
        learning_starts=HYPERPARAMS["learning_starts"],
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=LOG_DIR, 
    )

    logger = SimpleLogger(os.path.join(LOG_DIR, "training_log.csv"))

    print(f"\nStarting TD3 training...")

    model.learn(
        total_timesteps=HYPERPARAMS["total_timesteps"],
        callback=logger,
        progress_bar=True,
    )

    model.save(f"{POLICY}_asteroid_avoid")
    print(f"Model saved as {POLICY}_asteroid_avoid.zip")

    # Plot episode returns and loss
    if logger.returns:
        plt.figure(figsize=(14, 5))

        # Returns
        plt.subplot(1, 3, 1)
        plt.plot(logger.returns, alpha=0.3, label='Episode Return')
        if len(logger.returns) > 100:
            smooth = np.convolve(
                logger.returns, np.ones(100) / 100, mode='valid'
            )
            plt.plot(range(99, len(logger.returns)), smooth, 'r', linewidth=2, label='Smoothed (100 ep)')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Return per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Critic loss
        plt.subplot(1, 3, 2)
        if logger.losses:
            plt.plot(logger.losses, color='orange', alpha=0.5)
        plt.xlabel('Episode')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss')
        plt.grid(True, alpha=0.3)
        
        # Win rate over time
        plt.subplot(1, 3, 3)
        window = 100
        win_threshold = 1500 
        win_rates = []
        for i in range(window, len(logger.returns)):
            recent = logger.returns[i-window:i]
            wins = sum(1 for r in recent if r > win_threshold)
            win_rates.append(wins / window)
        plt.plot(range(window, len(logger.returns)), win_rates, 'g', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.title(f'Win Rate (last {window} episodes)')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, "training_plots.png"), dpi=150)

        print(f"\nTraining Complete!")
        print(f"Episodes: {len(logger.returns)}")
        if len(logger.returns) >= 100:
            final_mean = np.mean(logger.returns[-100:])
            final_wins = sum(1 for r in logger.returns[-100:] if r > win_threshold)
            final_win_rate = final_wins / 100
            print(f"Final mean (last 100): {final_mean:.1f}")
            print(f"Final win rate (last 100): {final_win_rate:.1%}")
        print(f"Best return: {max(logger.returns):.1f}")
        print(f"Plots saved to {LOG_DIR}")

    env.close()


if __name__ == "__main__":
    main()