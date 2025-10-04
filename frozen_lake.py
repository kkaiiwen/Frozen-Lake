import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# For debugging or making comparison
# np.random.seed(0)

# Environment
env = gym.make("FrozenLake-v1", map_name = "8x8", is_slippery = True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Lake map
SIZE = int(np.sqrt(n_states))
lake_map = np.array(env.unwrapped.desc, dtype = 'U1')

# Q-learning parameters
Q = np.zeros((n_states, n_actions))
alpha = 0.2
gamma = 0.95
epsilon = 1.0           # initial exploration
epsilon_min = 0.05      # floor
decay_rate = 0.00075    # exponential decay constant
episodes = 10000

# Convert state to (row, col)
def state_to_pos(state):
    return state // SIZE, state % SIZE

# Track metrics
reward_per_episode = []
max_Q_per_episode = []

# Train Q-learning and store last successful attempt
last_successful = []

for ep in range(episodes):
    state, info = env.reset()    # state, info = env.reset(seed = 0)
    done = False
    attempt_positions = [state_to_pos(state)]
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            best = np.flatnonzero(Q[state] == Q[state].max())
            action = np.random.choice(best)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] += alpha * (reward + gamma * (1 - int(done)) * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        attempt_positions.append(state_to_pos(state))
        total_reward += reward

    reward_per_episode.append(total_reward)
    max_Q_per_episode.append(np.max(Q))

    if total_reward > 0:
        last_successful = attempt_positions

    epsilon = max(epsilon_min, epsilon * np.exp(-decay_rate))

# Animation of last successful path
fig, ax = plt.subplots(figsize = (6, 6))
ax.set_xticks(np.arange(-0.5, SIZE, 1))
ax.set_yticks(np.arange(-0.5, SIZE, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)
ax.set_xlim(-0.5, SIZE - 0.5)
ax.set_ylim(-0.5, SIZE - 0.5)
ax.invert_yaxis()

# Draw door
def draw_door(ax, cx, cy):
    # Door rectangle (brown)
    door = Rectangle((cx - 0.20, cy - 0.25), 0.40, 0.55, facecolor = "#45322cff", edgecolor = 'black', linewidth = 1.5)
    ax.add_patch(door)
    # Knob (small yellow circle on right side)
    knob = Circle((cx + 0.12, cy + 0.02), 0.04, facecolor = '#fbc02d', edgecolor = 'black', linewidth = 1)
    ax.add_patch(knob)
    return [door, knob]

# Draw treasure chest
def draw_chest(ax, cx, cy):
    # Base box
    base = Rectangle((cx - 0.22, cy - 0.05), 0.44, 0.30, facecolor = '#8d6e63', edgecolor = 'black', linewidth = 1.5)
    # Lid
    lid  = Rectangle((cx - 0.22, cy - 0.35), 0.44, 0.12, facecolor = '#6d4c41', edgecolor = 'black', linewidth = 1.5, angle = 15)
    # Band + lock
    band = Rectangle((cx - 0.02, cy - 0.05), 0.04, 0.30, facecolor = '#fbc02d', edgecolor = 'black', linewidth = 1)
    lock = Rectangle((cx - 0.02, cy + 0.08), 0.04, 0.06, facecolor = '#fdd835', edgecolor = 'black', linewidth = 1)
    for p in (base, lid, band, lock):
        ax.add_patch(p)
    return [base, lid, band, lock]

# Draw lake tiles
for r in range(SIZE):
    for c in range(SIZE):
        tile = lake_map[r, c]
        color = 'royalblue' if tile == 'F' else 'green' if tile == 'S' else 'gold' if tile == 'G' else 'black'
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color = color))
        if tile == 'S':
            draw_door(ax, c, r)
        elif tile == 'G':
            draw_chest(ax, c, r)

# Draw robot
def create_robot(ax):
    # Body
    body = Rectangle((-1, -1), 0.36, 0.36, facecolor = '#90caf9', edgecolor = 'black', linewidth = 1.5)
    # Eyes
    eye_l = Circle((-1, -1), 0.03, facecolor = 'white', edgecolor = 'black', linewidth = 0.8)
    eye_r = Circle((-1, -1), 0.03, facecolor = 'white', edgecolor = 'black', linewidth = 0.8)
    # Add to axes
    ax.add_patch(body); ax.add_patch(eye_l); ax.add_patch(eye_r)
    return {"body": body, "eye_l": eye_l, "eye_r": eye_r}

# Move robot
def move_robot(robot, cx, cy):
    # Body is a Rectangle defined by lower-left, place so it's centered on (cx, cy)
    size = 0.36
    robot["body"].set_xy((cx - size/2, cy - size/2))
    # Eyes relative to center
    robot["eye_l"].center = (cx - 0.08, cy + 0.05)
    robot["eye_r"].center = (cx + 0.08, cy + 0.05)

# Robot agent animation
robot = create_robot(ax)
for row, col in last_successful:
    move_robot(robot, col, row)
    plt.pause(0.3)
if not last_successful:
    plt.title("No successful episode yet.")
plt.show()

# Q-map heatmaps
plt.figure(figsize = (8, 7))
action_names = ['Left', 'Down', 'Right', 'Up']

for action in range(n_actions):
    plt.subplot(2, 2, action + 1)
    q_values = Q[:, action].reshape(SIZE, SIZE)
    im = plt.imshow(q_values, cmap = 'viridis', origin = 'upper',
                    vmin = np.min(q_values), vmax = np.max(q_values))
    for r in range(SIZE):
        for c in range(SIZE):
            plt.text(c, r, f"{q_values[r, c]:.2f}", ha = 'center', va = 'center', 
                     color = 'white' if q_values[r, c] < np.max(q_values) / 2 else 'black', fontsize = 8)
    plt.colorbar(im)
    plt.title(f"Action: {action_names[action]}")
plt.tight_layout()
plt.show()

# Line graphs for key metrics
plt.figure(figsize = (12, 4))
plt.plot(reward_per_episode, label = 'Reward per Episode', color = 'blue')
plt.plot(max_Q_per_episode, label = 'Max Q-value', color = 'orange')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Q-learning Key Metrics Over Episodes')
plt.legend()
plt.show()

# Policy arrows map
action_symbols = ['←', '↓', '→', '↑']  # Left, Down, Right, Up

fig, ax = plt.subplots(figsize = (6, 6))
ax.set_xticks(np.arange(-0.5, SIZE, 1))
ax.set_yticks(np.arange(-0.5, SIZE, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)
ax.set_xlim(-0.5, SIZE - 0.5)
ax.set_ylim(-0.5, SIZE - 0.5)
ax.invert_yaxis()

# Draw tiles
for r in range(SIZE):
    for c in range(SIZE):
        tile = lake_map[r, c]
        color = 'royalblue' if tile == 'F' else 'green' if tile == 'S' else 'gold' if tile == 'G' else 'black'
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color = color))
        ax.text(c, r, tile, ha = 'center', va = 'center', 
                color = 'white' if tile in ['F', 'H'] else 'black', fontsize = 14)

# Draw policy arrows
for state in range(n_states):
    r, c = state_to_pos(state)
    if lake_map[r, c] not in ['H', 'G', 'S']:  # skip holes, start, goal
        best_action = np.argmax(Q[state])
        ax.text(c, r, action_symbols[best_action], ha = 'center', va = 'center', color = 'black', fontsize = 20)

plt.title("Optimal Policy Arrows Map")
plt.show()
