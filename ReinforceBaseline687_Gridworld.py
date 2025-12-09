
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from hyperparameter_tuning import run_grid_search

ACTIONS = ['U', 'D', 'L', 'R'] 
directions = {'U': (-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}


intended_rights = {'U': 'R', 'D': 'L', 'L': 'U', 'R': 'D'}
intended_lefts = {'U': 'L', 'D': 'R', 'L': 'D', 'R': 'U'}   

obstacles = {(2,2), (3,2)} 

REWARDS = {(4,4): 10, (4,2): -10} 
terminal_states = {(4,4)}

gamma = 0.925
all_states = set((x,y) for x in range(5) for y in range(5))
allowed_states = list(all_states - obstacles)

#Helper Functions
def print_pretty_grid(grid):
    print("\n--- Learned Policy ---")
    for r in range(5):
        row = ""
        for c in range(5):
            cell = grid[r][c]
            if (r, c) in obstacles:
                row += f"{'O':>8}" # Wall
            elif (r, c) in terminal_states:
                row += f"{'G':>8}"
            elif (r, c) == (4,2):
                if isinstance(cell, str): 
                    row += f"{cell+'(W)':>8}"
            elif isinstance(cell, float):
                row += f"{cell:8.4f}"  
            elif cell == 'N':
                row += f"{' ':>8}"
            else:
                row += f"{cell:>8}"
        print(row)

def get_next_state(s, a):
    p = np.random.random()

    if p < 0.80:
        # Moves in specified direction
        s_p = (s[0] + directions[a][0], s[1] + directions[a][1])
    elif p < 0.85: # Next 5%
        # Moves right with respect to intended direction
        right_dir = intended_rights[a]
        s_p = (s[0] + directions[right_dir][0], s[1] + directions[right_dir][1])
    elif p < 0.90: # Next 5%
        # Moves left with respect to intended direction
        left_dir = intended_lefts[a]
        s_p = (s[0] + directions[left_dir][0], s[1] + directions[left_dir][1])
    else:          # Remaining 10%
        # Stay in same state (robot breaks temporarily)
        s_p = s

    return s_p

def softmax(theta, state):
    x, y = state
    theta_values = theta[x,y] - np.max(theta[x,y])
    numerator = np.exp(theta_values)
    denominator = np.sum(numerator)

    return numerator/denominator

def sample_action(theta, state):
    probs = softmax(theta, state)
    return np.random.choice(ACTIONS, p=probs)


def generate_episode(theta):
    states = []
    actions = []
    rewards = []
    state = (0,0)

    while state not in terminal_states:
        action = sample_action(theta, state)
        next_state = get_next_state(state, action)
        if next_state not in allowed_states: # never enter forbidden states or coordinate outside of grid
            next_state = state
        x_p, y_p = next_state
        reward = 0 if next_state not in REWARDS else REWARDS[next_state]
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    
    return states, actions, rewards

def get_greedy_policy(theta):
    policy = np.empty((5,5), dtype=object)

    for x in range(5):
        for y in range(5):
            if (x,y) in obstacles:
                policy[x,y] = 'N'
            elif (x,y) in terminal_states:
                policy[x,y] = 'G'
            else:
                policy[x,y] = ACTIONS[np.argmax(theta[x,y])]
    
    return policy

def create_plot(returns, V_final):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PLOT 1: Learning Curve
    returns = np.array(returns)
    window = 100 # Larger window for smoother plot
    if len(returns) >= window:
        ma = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax1.plot(ma)
    else:
        ax1.plot(returns)
    ax1.set_xlabel("Episode (smoothed)")
    ax1.set_ylabel("Return (moving avg)")
    ax1.set_title("REINFORCE Training on 687-Gridworld")
    ax1.grid(True)

    # PLOT 2: Value Function Heatmap
    im = ax2.imshow(V_final, cmap='coolwarm', origin='upper')
    for i in range(5):
        for j in range(5):
            if (i,j) in obstacles:
                ax2.text(j, i, "WALL", ha="center", va="center", color="black", fontsize=8)
            elif (i,j) == (4,2):
                ax2.text(j, i, f"WATER\n{V_final[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
            else:
                ax2.text(j, i, f"{V_final[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    ax2.set_title("Learned State-Value Function V(s)")
    fig.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()






# ALGORITHMIC FUNCTIONS
def REINFORCE_WITH_BASELINE(num_episodes, alpha_w, alpha_theta):
    theta = np.zeros((5,5,4)) #policy parameterization
    V = np.zeros((5,5)) # state value function
    episode_returns = [] 
    episode_lengths = []
    

    for i in range(num_episodes):
        states, actions, rewards = generate_episode(theta)
        #print(states, actions, rewards)
        # Track metrics
        total_reward = sum(rewards)
        episode_returns.append(total_reward)
        episode_lengths.append(len(states))

        G = 0
        for t in reversed(range(len(states))):
            G = rewards[t] + gamma*G
            x,y = states[t]
            advantage = G - V[x,y]
            grad_V = 1 # as $\hat{v}(s, \mathbf{w})$ is just a lookup table where every state has its own independent weight.
            V[x,y] = V[x,y] + alpha_w * (gamma ** t) * advantage * grad_V
            probs = softmax(theta, states[t])
            # gradient of log-softmax: (1 - p) for chosen action and (-p) for others
            grad_ln_pi = -probs
            grad_ln_pi[ACTIONS.index(actions[t])] += 1.0
            theta[x,y] = theta[x,y] + alpha_theta * (gamma ** t) * advantage * grad_ln_pi
    
    return episode_returns, episode_lengths, V, get_greedy_policy(theta)


if __name__ == "__main__":
    returns, lengths, V_final, policy = REINFORCE_WITH_BASELINE(
        num_episodes=5_000, 
        alpha_w=0.1, 
        alpha_theta=0.005 
    )
    print_pretty_grid(policy)

    create_plot(returns, V_final)
    #run_grid_search()
