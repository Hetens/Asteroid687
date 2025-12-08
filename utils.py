import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


class SimpleLogger(BaseCallback):
    """Common logger for all RL algorithms"""
    def __init__(self, csv_path, loss_key='train/loss'):
        super().__init__()
        self.csv_path = csv_path
        self.loss_key = loss_key  # Different algos use different loss keys
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

                # Try multiple possible loss keys
                loss = 0.0
                loss_keys = [self.loss_key, 'train/critic_loss', 'train/value_loss', 'train/loss']
                for key in loss_keys:
                    if key in self.logger.name_to_value:
                        loss = self.logger.name_to_value[key]
                        break

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


def save_hyperparams(log_dir, policy_name, n_envs, hyperparams):
    """Save hyperparameters to a text file"""
    path = os.path.join(log_dir, "hyperparams.txt")
    with open(path, 'w') as f:
        f.write(f"Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{policy_name} Hyperparameters:\n")
        f.write(f"Number of parallel environments: {n_envs}\n")
        f.write("-" * 30 + "\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")


def plot_training_results(logger, log_dir, loss_label='Loss'):
    """Generate training plots"""
    if not logger.returns:
        return

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

    # Loss
    plt.subplot(1, 3, 2)
    if logger.losses:
        plt.plot(logger.losses, color='orange', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel(loss_label)
    plt.title(loss_label)
    plt.grid(True, alpha=0.3)
    
    # Win rate over time
    plt.subplot(1, 3, 3)
    window = 100
    win_threshold = 2000 
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
    plt.savefig(os.path.join(log_dir, "training_plots.png"), dpi=150)


def print_training_summary(logger, policy_name):
    """Print final training statistics"""
    print(f"\nTraining Complete!")
    print(f"Episodes: {len(logger.returns)}")
    
    if len(logger.returns) >= 100:
        win_threshold = 1500
        final_mean = np.mean(logger.returns[-100:])
        final_wins = sum(1 for r in logger.returns[-100:] if r > win_threshold)
        final_win_rate = final_wins / 100
        print(f"Final mean (last 100): {final_mean:.1f}")
        print(f"Final win rate (last 100): {final_win_rate:.1%}")
    
    print(f"Best return: {max(logger.returns):.1f}")
    print(f"Model saved as {policy_name}_asteroid_avoid.zip")