import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from asteroid_env import AsteroidAvoidEnv

LOG_DIR = "./logs/"
os.makedirs(LOG_DIR, exist_ok=True)
np.set_printoptions(precision=4, suppress=True)
# Hyperparameters
HYPERPARAMS = {
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.1,
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
            csv.writer(f).writerow(['episode', 'return', 'length', 'value_loss'])
    
    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.last_ep_count += 1
                ep_return = info['episode']['r']
                ep_length = info['episode']['l']
                loss = self.logger.name_to_value.get('train/value_loss', 0.0)
                
                self.returns.append(ep_return)
                self.losses.append(loss)
                
                with open(self.csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([self.last_ep_count, ep_return, ep_length, loss])
                
                if self.last_ep_count % 100 == 0:
                    print(f"Ep {self.last_ep_count}: Mean Return = {np.mean(self.returns[-100:]):.1f}")
        return True


def save_hyperparams(path):
    """Save hyperparameters to a txt file."""
    with open(path, 'w') as f:
        f.write(f"Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        f.write("PPO Hyperparameters:\n")
        f.write("-"*30 + "\n")
        for key, value in HYPERPARAMS.items():
            f.write(f"{key}: {value}\n")
    print(f"Hyperparameters saved to: {path}")


def main():
    print("Training PPO on Asteroid Avoid...")
    
    # Save hyperparameters
    save_hyperparams(os.path.join(LOG_DIR, "hyperparams.txt"))
    
    env = Monitor(AsteroidAvoidEnv())
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
        verbose=1,
    )

    logger = SimpleLogger(os.path.join(LOG_DIR, "training_log.csv"))
    model.learn(total_timesteps=HYPERPARAMS["total_timesteps"], callback=logger, progress_bar=True)
    model.save("ppo_asteroid_avoid")
    
    # Plot return vs episode
    if logger.returns:
        plt.figure(figsize=(8, 5))
        plt.plot(logger.returns, alpha=0.4)
        if len(logger.returns) > 20:
            smooth = np.convolve(logger.returns, np.ones(20)/20, mode='valid')
            plt.plot(range(19, len(logger.returns)), smooth, 'r', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Return per Episode')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR, "return_plot.png"), dpi=150)
        plt.close()
        
        # Plot loss vs episode
        plt.figure(figsize=(8, 5))
        plt.plot(logger.losses, color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Value Loss')
        plt.title('Value Loss per Episode')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR, "loss_plot.png"), dpi=150)
        plt.close()
        
        print(f"\nDone! Episodes: {len(logger.returns)}, Final mean: {np.mean(logger.returns[-100:]):.1f}")
        print(f"Plots saved to {LOG_DIR}")
    
    env.close()


if __name__ == "__main__":
    main()
