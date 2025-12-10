import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from ReinforceBaseline687_Gridworld import REINFORCE_WITH_BASELINE


def save_results(returns, V_final, policy_grid, filename_base, params_str):
    """
    Saves the plot and the policy text to a file.
    """
    # 1. Save Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Smooth returns
    returns = np.array(returns)
    window = 100
    if len(returns) >= window:
        ma = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax1.plot(ma)
    else:
        ax1.plot(returns)
        
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return (Moving Avg)")
    ax1.set_title(f"Training: {params_str}")
    ax1.grid(True)

    im = ax2.imshow(V_final, cmap='coolwarm', origin='upper')
    for i in range(5):
        for j in range(5):
            if (i,j) in obstacles:
                ax2.text(j, i, "WALL", ha="center", va="center", color="black", fontsize=8)
            elif (i,j) == (4,2):
                ax2.text(j, i, f"WATER\n{V_final[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
            else:
                ax2.text(j, i, f"{V_final[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    ax2.set_title("Learned V(s)")
    fig.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig(filename_base + ".png")
    plt.close() # Close to free memory

    with open(filename_base + ".txt", "w") as f:
        f.write(f"Parameters: {params_str}\n")
        f.write("-" * 40 + "\n")
        for r in range(5):
            row_str = ""
            for c in range(5):
                row_str += f"{policy_grid[r][c]:>8}"
            f.write(row_str + "\n")

def run_grid_search():
    alphas_theta = [0.001, 0.005, 0.01] 
    alphas_w = [0.05, 0.1, 0.2]
    
    num_episodes = 5000 
    
    # Create output directory
    output_dir = "grid_search_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"{'Alpha Theta':<15} | {'Alpha W':<15} | {'Final Avg Return':<20} | {'Status'}")
    print("-" * 70)

    all_results = {}

    for a_theta, a_w in itertools.product(alphas_theta, alphas_w):
        params_key = f"th_{a_theta}_w_{a_w}"
        filename = os.path.join(output_dir, f"run_{params_key}")
        
        # Run Algorithm
        returns, lengths, V, policy = REINFORCE_WITH_BASELINE(num_episodes, a_w, a_theta)
        
        window = 100
        ma = np.convolve(returns, np.ones(window)/window, mode='valid')
        final_score = ma[-1] if len(ma) > 0 else 0
        
        save_results(returns, V, policy, filename, f"a_theta={a_theta}, a_w={a_w}")
        
        print(f"{a_theta:<15} | {a_w:<15} | {final_score:<20.4f} | Saved")
        
        downsampled_ma = ma[::50].tolist()
        all_results[params_key] = downsampled_ma

    print("\n\n" + "="*80)
    print("DATA FOR ANALYSIS (Copy and paste this below):")
    print("="*80)
    for key, data in all_results.items():
        print(f"'{key}': {data},")