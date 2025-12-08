
import numpy as np
import os
import matplotlib.pyplot as plt
 
ACTIONS = ['U', 'D', 'L', 'R'] 
directions = {'U': (-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}
intended_rights = {'U': 'R', 'D': 'L', 'L': 'U', 'R': 'D'}
intended_lefts = {'U': 'L', 'D': 'R', 'L': 'D', 'R': 'U'}   
forbidden_furniture = {(2,1), (2,2), (2,3), (3,2)} #same behaviour for cells beyond walls
REWARDS = {(4,4): 10, (0,3): -8, (4,1): -8} #if not in this then -0.05
terminal_states = {(4,4)}
gamma = 0.925
all_states = set((x,y) for x in range(5) for y in range(5))
allowed_states = list(all_states - forbidden_furniture)
V_star = np.array([[2.6638, 2.9969, 2.8117, 3.6671, 4.8497],
            [2.9712, 3.5101, 4.0819, 4.8497, 7.1648],
            [2.5935, 0.     , 0.     , 0.     , 8.4687],
            [2.0992, 1.0849, 0.     , 8.6097, 9.5269],
            [1.0849, 4.9465, 8.4687, 9.5269, 0.    ]])
pi_star = [['R', 'D', 'L', 'D', 'D'], ['R', 'R', 'R', 'R', 'D'], ['U', 'N', 'N', 'N', 'D'], ['U', 'L', 'N', 'D', 'D'], ['U', 'R', 'R', 'R', 'G']]


#Helper Functions
def print_pretty_grid(grid):
    for r in range(len(grid)):
        row = ""
        for c in range(len(grid[0])):
            cell = grid[r][c]
            if isinstance(cell, float):
                row += f"{cell:8.4f}"  
            elif (r, c) in terminal_states:
                row += f"{'G':>8}"
            elif cell == 'N':
                row += f"{' ':>8}"
            else:
                row += f"{cell:>8}"
        print(row)

def get_next_state(s, a):
    p = np.random.random()

    if p < 0.7:
        #Moves in specified direction
        s_p = (s[0] + directions[a][0], s[1] + directions[a][1])
    elif p < 0.82: #0.12
        #Moves right with respect to intended direction
        right_dir = intended_rights[a]
        s_p = (s[0] + directions[right_dir][0], s[1] + directions[right_dir][1])
    elif p < 0.94:          # next 12%
        #Moves left with respect to intended direction
        left_dir = intended_lefts[a]
        s_p = (s[0] + directions[left_dir][0], s[1] + directions[left_dir][1])
    else:                  # last 6%: stay in same state
        #Stay in same state
        s_p = s

    return s_p

def sample_initial_state():
    sampling_states = list(all_states - forbidden_furniture - terminal_states)
    return sampling_states[np.random.choice(len(sampling_states))]

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
    state = sample_initial_state()

    while state not in terminal_states:
        action = sample_action(theta, state)
        next_state = get_next_state(state, action)
        if next_state not in allowed_states: # never enter forbidden states or coordinate outside of grid
            next_state = state
        x_p, y_p = next_state
        reward = -0.05 if next_state not in REWARDS else REWARDS[next_state]
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    
    return states, actions, rewards

def get_greedy_policy(theta):
    policy = np.empty((5,5), dtype=object)

    for x in range(5):
        for y in range(5):
            if (x,y) in forbidden_furniture:
                policy[x,y] = 'N'
            elif (x,y) in terminal_states:
                policy[x,y] = 'G'
            else:
                policy[x,y] = ACTIONS[np.argmax(theta[x,y])]
    
    return policy

def create_plot(returns, V_final):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #PLOT 1
    #: Learning Curve (with Moving Average)
    #Calculate a rolling average to smooth out the noise
    returns = np.array(returns)
    window = 50
    ma = np.convolve(returns, np.ones(window)/window, mode='valid')

    ax1.plot(ma)
    ax1.set_xlabel("Episode (smoothed)")
    ax1.set_ylabel("Return (moving avg)")
    ax1.set_title("REINFORCE training")
    ax1.grid(True)

    #PLOT 2
    #Value Function Heatmap
    #Red = Good, Blue = Bad
    im = ax2.imshow(V_final, cmap='coolwarm', origin='upper')

    # Add text annotations for the values
    for i in range(5):
        for j in range(5):
            if (i,j) in forbidden_furniture:
                text = ax2.text(j, i, "WALL", ha="center", va="center", color="black", fontsize=8)
            else:
                text = ax2.text(j, i, f"{V_final[i, j]:.2f}", 
                                ha="center", va="center", color="black", fontsize=8)

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

# 1. Run the training
returns, lengths, V_final, policy = REINFORCE_WITH_BASELINE(
    num_episodes=10_000, 
    alpha_w=0.1, 
    alpha_theta=0.005 # Lower alpha usually helps REINFORCE stability
)
print_pretty_grid(policy)

create_plot(returns, V_final)