import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from asteroid_env import AsteroidAvoidEnv

LOG_DIR = "./ppo/logs/"
os.makedirs(LOG_DIR, exist_ok=True)

POLICY = "PPO"
N_ENVS = 16

HYPERPARAMS = {
    "total_timesteps": 10_000_000,
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
                if 'train/value_loss' in self.logger.name_to_value:
                    loss = self.logger.name_to_value['train/value_loss']
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
        device=device,
        tensorboard_log=LOG_DIR, 
    )

    logger = SimpleLogger(os.path.join(LOG_DIR, "training_log.csv"))

    print(f"\n🚀 Starting PPO training...")
    print(f"Tips: Watch for WinRate in logs. Target: 50%+ for good performance")

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

        # Value loss
        plt.subplot(1, 3, 2)
        if logger.losses:
            plt.plot(logger.losses, color='orange', alpha=0.5)
        plt.xlabel('Episode')
        plt.ylabel('Value Loss')
        plt.title('Value Loss')
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