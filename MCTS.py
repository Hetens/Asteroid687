import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok= True)
os.makedirs('results', exist_ok= True)
np.set_printoptions(precision = 4, suppress=True)


###### INITIALIZATION ###########
grid_size = 5
num_actions = 4
DISCOUNT = 0.925
NUM_EPISODES = 10000

actions = [
    [0, 1],  # 0: AR
    [0, -1], # 1: AL
    [1, 0],  # 2: AD
    [-1, 0]  # 3: AU
]

#constant R(s,a,s') and forbidden furniture at 0 as a placeholder
rewards = [[-0.05, -0.05, -0.05, -8.0, -0.05],
           [-0.05, -0.05, -0.05, -0.05, -0.05],
           [-0.05, 0.0, 0.0, 0.0,-0.05],
           [-0.05, -0.05, 0.0, -0.05, -0.05],
           [-0.05, -8.0, -0.05, -0.05, 10.0]]
rewards = np.array(rewards, dtype = np.float32)
#value_initially
v_i = np.zeros((grid_size, grid_size), dtype=np.float32)

#q value initially
q_value = np.zeros((grid_size, grid_size, num_actions), dtype=np.float32)

#p(s,a,s')
environ_probs = [0.7, 0.12, 0.12, 0.06]

initial_state = [0,0] # monsters = [3,0],[4,1] #food  = [4,4] # 
furniture = [[2,1], [2,2], [2,3], [3,2]]

best_actions = np.zeros((grid_size, grid_size), dtype=np.float32)
#Initially lets start with a random policy which means in each state i have 4 probs,


# v* ------------------------------------------------------------------------------- HW3
opt_values = np.array([[2.6638, 2.9969, 2.8117, 3.6671, 4.8497],
            [2.9713, 3.5101, 4.0819, 4.8497, 7.1648],
            [2.5936, 0.0,     0.0,     0.0,     8.4687],
            [2.0992, 1.0849,  0.0,     8.6097,  9.5269],
            [1.0849, 4.9465, 8.4687, 9.5269, 0.0    ]])

best_actions = np.array([
    [0, 2, 1, 2, 2],
    [0, 0, 0, 0, 2],
    [3, 0, 0, 0, 2],
    [3, 1, 0, 0, 2],
    [3, 0, 0, 0, 0]
], dtype=np.int32)

#OPTIMAL POLICY -------------------------------------------------------------------------

best_policy = np.eye(num_actions)[best_actions]
# policy = best_policy

terminal_states = [[4,4]]

visit_counts = defaultdict(int)
visit_counts_q = defaultdict(int)
# returns = defaultdict(list)
# returns_q = defaultdict(list)

valid_start_states = []
for i in range(grid_size):
    for j in range(grid_size):
        if [i, j] not in furniture and [i, j] not in terminal_states:
            valid_start_states.append([i, j])

####### INITIALIZATION COMPLETE ###########

def reachable_states(state, action_index):
    # 0=Right, 1=Left, 2=Down, 3=Up
    
    action_outcomes = {
        0: [0, 2, 3, None],  # Right: intended=right, right=down, left=up, stay=same
        1: [1, 3, 2, None],  # Left: intended=left, right=up, left=down, stay=same
        2: [2, 1, 0, None],  # Down: intended=down, right=left, left=right, stay=same
        3: [3, 0, 1, None],  # Up: intended=up, right=right, left=left, stay=same
    }
    
    transition_probs = [0.70, 0.12, 0.12, 0.06]  # [intended, right, left, stay]
    
    outcome_idx = np.random.choice(len(transition_probs), p=transition_probs)
    outcome_action_idx = action_outcomes[action_index][outcome_idx]

    if outcome_action_idx is None:  
        ni, nj = state[0], state[1]
    else:
        x, y = actions[outcome_action_idx]
        ni, nj = state[0] + x, state[1] + y

        # Check boundaries and furniture
        if (
            ni < 0 or ni >= grid_size or
            nj < 0 or nj >= grid_size or
            [ni, nj] in furniture
        ):
            ni, nj = state[0], state[1] 
    
    
    return [ni, nj], rewards[ni][nj]


# MCTS helpers
Q_mcts = defaultdict(lambda: np.zeros(num_actions))
N_mcts = defaultdict(lambda: np.zeros(num_actions))


def uct_select(state):
    state_tuple = tuple(state)
    best_action = -1
    best_score = -float('inf')
    
    # Total visits to this state
    N_s = np.sum(N_mcts[state_tuple])
    
    for a in range(num_actions):
        if N_mcts[state_tuple][a] == 0:
            # If action not sampled yet, treat as high priority or handle in expansion
            return a
        
        q_val = Q_mcts[state_tuple][a]
        # UCT formula: Q(s,a) + C * sqrt(ln(N(s)) / N(s,a))
        # Increasing C to match reward scale (approx -10 to 10)
        c_param = 15.0
        exploration = c_param * np.sqrt(np.log(N_s) / N_mcts[state_tuple][a])
        score = q_val + exploration
        
        if score > best_score:
            best_score = score
            best_action = a
            
    return best_action


def mcts_iteration():
    # 1. Selection & Expansion
    # Start at a random valid state or a fixed root?
    # To solve the whole grid, starting random is good (like solving all subgames)
    start_node = valid_start_states[np.random.randint(len(valid_start_states))]
    curr = start_node.copy()
    
    path = [] # Stores (state, action, reward)
    depth = 0
    max_depth = 25 # Prevent infinite loops in cycles
    
    # Selection: Traverse until we hit a terminal or an unexpanded node
    while depth < max_depth:
        if curr in terminal_states:
            break
            
        state_tuple = tuple(curr)
        
        # Check if fully expanded
        unexpanded_actions = [a for a in range(num_actions) if N_mcts[state_tuple][a] == 0]
        
        if unexpanded_actions:
            # Expansion: Pick one untried action, step, and then rollout
            action = np.random.choice(unexpanded_actions)
            next_state, reward = reachable_states(curr, action)
            path.append((curr, action, reward))
            curr = next_state
            break # Go to rollout
        else:
            # All actions tried, use UCT to go deeper
            action = uct_select(curr)
            next_state, reward = reachable_states(curr, action)
            path.append((curr, action, reward))
            curr = next_state
            depth += 1
            # Continue selection loop
            
    # 2. Simulation (Rollout)
    # Run random policy until terminal or max depth
    rollout_return = 0
    rollout_depth = 0
    rollout_curr = curr
    gamma_pow = 1.0
    
    # We collect return from the rollout part
    # Note: The 'path' rewards are handled in backprop. We need G from the leaf.
    
    while rollout_curr not in terminal_states and rollout_depth < 100:
        # Random action
        action = np.random.randint(num_actions)
        next_s, r = reachable_states(rollout_curr, action)
        
        rollout_return += gamma_pow * r
        gamma_pow *= DISCOUNT
        rollout_curr = next_s
        rollout_depth += 1
        
    # 3. Backpropagation
    # Update Q and N for nodes in the path
    # G_node = r_immediate + gamma * r_next ...
    # We can effectively compute G backwards
    
    G = rollout_return # The value from the leaf onwards
    
    for i in reversed(range(len(path))):
        s_node, a_node, r_node = path[i]
        state_tuple = tuple(s_node)
        
        G = r_node + DISCOUNT * G
        
        N_mcts[state_tuple][a_node] += 1
        n = N_mcts[state_tuple][a_node]
        q = Q_mcts[state_tuple][a_node]
        
        # Incremental mean update
        Q_mcts[state_tuple][a_node] = q + (G - q) / n

def calc_mse(q_table, optimal_v):
    num_states = 0.0
    mse_sum = 0.0

    # Derive V(s) from Q(s, a) as max_a Q(s,a)
    for i in range(grid_size):
        for j in range(grid_size):
            if [i, j] not in furniture and [i, j] not in terminal_states:
                num_states += 1
                v_est = np.max(q_table[i, j]) if np.sum(N_mcts[(i,j)]) > 0 else 0.0
                mse_sum += (optimal_v[i, j] - v_est) ** 2
    
    return mse_sum / num_states if num_states > 0 else 0.0


def evaluate_policy(num_eval_episodes=10):
    total_return = 0.0
    
    for _ in range(num_eval_episodes):
        # Start at a random valid start state
        state = valid_start_states[np.random.randint(len(valid_start_states))].copy()
        episode_return = 0.0
        output_gamma = 1.0
        steps = 0
        
        while steps < 100:
            if state in terminal_states:
                break
                
            state_tuple = tuple(state)
            
            # Greedy action from Q_mcts
            # If state not visited, pick random
            if np.sum(N_mcts[state_tuple]) > 0:
                action = np.argmax(Q_mcts[state_tuple])
            else:
                action = np.random.choice(num_actions)
                
            next_state, reward = reachable_states(state, action)
            episode_return += output_gamma * reward
            output_gamma *= DISCOUNT
            state = next_state
            steps += 1
            
        total_return += episode_return
        
    return total_return / num_eval_episodes

def run_experiment_mcts():
    # Reset globals for new experiment
    Q_mcts.clear()
    N_mcts.clear()
    
    mse_values = []
    avg_returns = []
    episode_numbers = []
    
    # Iterations corresponding to "episodes" in previous code
    # Since MCTS is one trace per iteration, it's comparable
    for i in range(1, NUM_EPISODES + 1):
        mcts_iteration()
        
        if i % 250 == 0:
            # Convert default dict Q to numpy array for MSE calc
            q_grid = np.zeros((grid_size, grid_size, num_actions))
            for r in range(grid_size):
                for c in range(grid_size):
                    q_grid[r, c] = Q_mcts[(r, c)]
            
            mse = calc_mse(q_grid, opt_values)
            mse_values.append(mse)
            
            # Evaluate Average Return
            avg_ret = evaluate_policy()
            avg_returns.append(avg_ret)
            
            episode_numbers.append(i)
            
        if i % 100 == 0:
            print(f"MCTS Iteration: {i}")

    # Build final V and Policy
    final_values = np.zeros((grid_size, grid_size))
    final_policy = np.zeros((grid_size, grid_size, num_actions))
    
    for i in range(grid_size):
        for j in range(grid_size):
            final_values[i, j] = np.max(Q_mcts[(i, j)])
            best_a = np.argmax(Q_mcts[(i, j)])
            final_policy[i, j, best_a] = 1.0
            
    return episode_numbers, mse_values, avg_returns, final_values, final_policy

results = {}

print("Starting MCTS Experiment...")
episodes, mse, avg_returns, values, policy = run_experiment_mcts()

name = "MCTS"

# SAVING PLOTS OF MSE FOR EACH RUN -----------------
plt.figure() 
plt.title(f"{name} MSE")
plt.plot(episodes, mse)
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.savefig(f"plots/{name}_mse.png")
plt.close() 

# SAVING PLOTS OF AVERAGE RETURN -----------------
plt.figure() 
plt.title(f"{name} Average Return")
plt.plot(episodes, avg_returns)
plt.xlabel("Iterations")
plt.ylabel("Average Return")
plt.savefig(f"plots/{name}_return.png")
plt.close() 

#PRINTING FOR EACH RUN -------
print(f"Algorithm {name} for {NUM_EPISODES} iterations.")
print("\nFinal Values:")
print(values)
print("\nOptimal Policy (Best Actions):")
arrow_map = {0: '→', 1: '←', 2: '↓', 3: '↑'}
arrow_grid = np.full((grid_size, grid_size), ' ', dtype='<U2')

for i in range(grid_size):
    for j in range(grid_size):
        if [i, j] in furniture:
            arrow_grid[i, j] = 'x'  
        elif [i, j] in terminal_states:
            arrow_grid[i, j] = 'T'
        else:
            # Check if visited
            if np.sum(N_mcts[(i,j)]) > 0:
                arrow_grid[i, j] = arrow_map[int(np.argmax(Q_mcts[(i, j)]))]
            else:
                arrow_grid[i, j] = '?'

print("\nPolicy Arrows:")
for row in arrow_grid:
    print(' '.join(row))

result_filename = f"results/{name}.txt"
with open(result_filename, 'w', encoding='utf-8') as f:
    f.write(f"Results for: {name}\n")
    f.write("\n\n")
    f.write("Final Values (v_i):\n")
    f.write(np.array2string(values, precision=4, suppress_small=True))
    f.write("\n\n")

    f.write("Policy Arrows:\n")
    for row in arrow_grid:
        f.write(' '.join(row) + '\n')
    
    f.write(f"\nFinal MSE: {mse[-1]:.4f}\n")
    f.write(f"Final Avg Return: {avg_returns[-1]:.4f}\n")

print(f"Saved results to {result_filename}")