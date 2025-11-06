import pygame
import random
import time
import json
from collections import deque, defaultdict
import os
import math

# =================== CONFIG ===================
GRID_SIZE = 50
CELL_SIZE = 12
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30
OBSTACLE_RATIO = 0.4
LIDAR_RADIUS = 5
MOVE_DELAY = 0.06     # time between steps for each robot
MAP_FILE = "shared_map.json"

NUM_ROBOTS = 6        # adjust number of swarm robots
MAX_EPISODE_STEPS = 5000  # safety cap for very long runs

# Q-Learning params
ALPHA = 0.6           # learning rate
GAMMA = 0.95          # discount factor
EPS_START = 0.9       # initial exploration
EPS_END = 0.05        # final exploration
EPS_DECAY = 0.9995    # per-step decay

# Rewards
REWARD_STEP = -0.1
REWARD_GOAL = 200.0
REWARD_INVALID = -5.0
REWARD_BLOCKED = -2.0

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (200, 200, 200)
COLORS = [RED, BLUE, GREEN, PURPLE, CYAN, MAGENTA, ORANGE, YELLOW]

GOAL_POS = (GRID_SIZE // 2, GRID_SIZE // 2)

# Actions: (dx,dy)
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]
ACTION_NAMES = ["L","R","U","D"]

# =================== GRID UTIL ===================
def generate_grid():
    """Generate a random grid where each start has a path to goal."""
    while True:
        grid = [[0 if random.random() > OBSTACLE_RATIO else 1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        grid[GOAL_POS[1]][GOAL_POS[0]] = 0
        starts = get_start_positions(NUM_ROBOTS)
        for sx, sy in starts:
            grid[sy][sx] = 0
        if all(path_exists(grid, s, GOAL_POS) for s in starts):
            return grid

def path_exists(grid, start, goal):
    queue = deque([start])
    visited = set([start])
    while queue:
        x, y = queue.popleft()
        if (x,y) == goal:
            return True
        for dx, dy in ACTIONS:
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny][nx] == 0 and (nx,ny) not in visited:
                visited.add((nx,ny))
                queue.append((nx,ny))
    return False

def get_start_positions(n):
    corners = [(0,0), (GRID_SIZE-1,0), (0,GRID_SIZE-1), (GRID_SIZE-1,GRID_SIZE-1)]
    edges = [(GRID_SIZE//2,0), (0,GRID_SIZE//2), (GRID_SIZE-1,GRID_SIZE//2), (GRID_SIZE//2,GRID_SIZE-1)]
    spots = corners + edges
    # Slight jitter to reduce exact overlap if many robots
    positions = []
    for i in range(n):
        sx, sy = spots[i % len(spots)]
        positions.append((sx, sy))
    return positions

# =================== SHARED MAP FILE ===================
def init_shared_map(robots):
    data = {
        "obstacles": [],
        "visited": [],
        "robots": {r.name: {"position": list(r.pos), "reached_goal": False} for r in robots}
    }
    with open(MAP_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_shared_map():
    if not os.path.exists(MAP_FILE):
        return {"obstacles": [], "visited": [], "robots": {}}
    with open(MAP_FILE, "r") as f:
        return json.load(f)

def save_shared_map(shared_map, robots):
    data = {
        "obstacles": sorted([list(p) for p in shared_map["obstacles"]]),
        "visited": sorted([list(p) for p in shared_map["visited"]]),
        "robots": {
            r.name: {"position": [r.pos[0], r.pos[1]], "reached_goal": r.reached_goal} for r in robots
        }
    }
    with open(MAP_FILE, "w") as f:
        json.dump(data, f, indent=2)

# =================== Q-Learning Agent (Robot) ===================
class Robot:
    def __init__(self, name, color, start, grid, shared_map):
        self.name = name
        self.color = color
        self.pos = start
        self.grid = grid
        self.shared_map = shared_map            # local sets: 'obstacles', 'visited', 'ready'
        self.reached_goal = False
        self.last_move_time = 0.0
        self.frontier = deque()
        self.visited = set()
        # Q-table as dict: state (x,y) -> [q0,q1,q2,q3]
        self.Q = defaultdict(lambda: [0.0 for _ in ACTIONS])
        # exploration
        self.epsilon = EPS_START

    def scan(self):
        """Return obstacles detected in LiDAR radius (local knowledge)."""
        obs = set()
        x0, y0 = self.pos
        for dx in range(-LIDAR_RADIUS, LIDAR_RADIUS+1):
            for dy in range(-LIDAR_RADIUS, LIDAR_RADIUS+1):
                nx, ny = x0+dx, y0+dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if self.grid[ny][nx] == 1:
                        obs.add((nx, ny))
        return obs

    def sense_and_publish(self):
        seen = self.scan()
        # update shared_map local sets
        self.shared_map['obstacles'].update(seen)
        self.shared_map['visited'].add(self.pos)
        self.shared_map['ready'][self.name] = True
        # write shared map file
        save_shared_map(self.shared_map, robots)  # robots is global list - safe here

    def choose_action(self):
        """Epsilon-greedy pick from Q-table, but prefer valid moves."""
        s = self.pos
        qvals = self.Q[s]
        # read live shared map to get known obstacles
        live = load_shared_map()
        known_obs = set(tuple(p) for p in live.get("obstacles", []))
        valid_actions = []
        for a_idx, (dx,dy) in enumerate(ACTIONS):
            nx, ny = s[0]+dx, s[1]+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx,ny) not in known_obs:
                valid_actions.append(a_idx)
        # if no valid actions, return None
        if not valid_actions:
            return None

        # exploration vs exploitation
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # pick best among valid actions
            best_idx = max(valid_actions, key=lambda i: qvals[i])
            return best_idx

    def step(self):
        """Perform one learning step: pick action, observe reward, update Q, move (if valid)."""
        if self.reached_goal:
            return

        # throttle moves
        if time.time() - self.last_move_time < MOVE_DELAY:
            return
        self.last_move_time = time.time()

        # sync and gather known obstacles/visited
        live = load_shared_map()
        known_obs = set(tuple(p) for p in live.get("obstacles", []))
        known_visited = set(tuple(p) for p in live.get("visited", []))

        s = self.pos
        self.visited.add(s)
        self.shared_map['visited'].add(s)

        action_idx = self.choose_action()
        if action_idx is None:
            # trapped (according to shared knowledge)
            # give small negative reward and set reached_goal True to stop further attempts
            self.reached_goal = True
            return

        dx, dy = ACTIONS[action_idx]
        nx, ny = s[0]+dx, s[1]+dy
        next_state = (nx, ny)

        # Determine reward and whether move is allowed
        invalid_move = False
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            reward = REWARD_INVALID
            next_state = s
            invalid_move = True
        elif (nx, ny) in known_obs:
            # knowledge-based blocked move
            reward = REWARD_BLOCKED
            next_state = s
            invalid_move = True
        elif self.grid[ny][nx] == 1:
            # actual obstacle (maybe undiscovered)
            self.shared_map['obstacles'].add((nx, ny))
            reward = REWARD_INVALID
            next_state = s
            invalid_move = True
        else:
            # valid move to free cell
            # step cost
            reward = REWARD_STEP
            # extra small penalty if moving to a cell already known visited (discourage redundancy)
            if (nx, ny) in known_visited:
                reward += -0.05

        # big reward for reaching goal
        if (nx, ny) == GOAL_POS:
            reward += REWARD_GOAL

        # Q-learning update
        s_key = s
        ns_key = next_state
        old_q = self.Q[s_key][action_idx]
        next_max = max(self.Q[ns_key])
        self.Q[s_key][action_idx] = old_q + ALPHA * (reward + GAMMA * next_max - old_q)

        # apply move if not invalid
        if not invalid_move:
            self.pos = next_state
            self.shared_map['visited'].add(self.pos)

        # if moved into goal
        if self.pos == GOAL_POS:
            self.reached_goal = True

        # decay exploration slightly
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

# =================== MAIN SIMULATION ===================
def reconstruct_and_display(shared_map, robots, screen):
    """Display reconstructed final map from shared_map file until user closes."""
    pygame.display.set_caption("Reconstructed Shared Map")
    data = load_shared_map()
    visited = set(tuple(p) for p in data.get("visited", []))
    obstacles = set(tuple(p) for p in data.get("obstacles", []))

    # draw once
    screen.fill(WHITE)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (x,y) in obstacles:
                pygame.draw.rect(screen, BLACK, rect)
            elif (x,y) in visited:
                pygame.draw.rect(screen, GRAY, rect)
    gx, gy = GOAL_POS
    pygame.draw.rect(screen, YELLOW, pygame.Rect(gx*CELL_SIZE, gy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for r in robots:
        rx, ry = r.pos
        pygame.draw.circle(screen, r.color, (rx*CELL_SIZE + CELL_SIZE//2, ry*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2)
    pygame.display.flip()

    print("Reconstructed map displayed. Close window to exit.")
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return

def main():
    global robots
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Swarm Q-Learning Exploration")
    clock = pygame.time.Clock()

    grid = generate_grid()
    shared_map = {'obstacles': set(), 'visited': set(), 'ready': {}}
    starts = get_start_positions(NUM_ROBOTS)
    robots = [
        Robot(f"R{i+1}", COLORS[i % len(COLORS)], starts[i], grid, shared_map)
        for i in range(NUM_ROBOTS)
    ]
    for r in robots:
        shared_map['ready'][r.name] = False

    init_shared_map(robots)

    paused = False
    countdown = 3.0
    start_time = None
    running = True
    step_count = 0
    episode_steps = 0

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # regenerate fresh random map and restart
                    grid = generate_grid()
                    shared_map = {'obstacles': set(), 'visited': set(), 'ready': {}}
                    starts = get_start_positions(NUM_ROBOTS)
                    robots = [
                        Robot(f"R{i+1}", COLORS[i % len(COLORS)], starts[i], grid, shared_map)
                        for i in range(NUM_ROBOTS)
                    ]
                    for r in robots:
                        shared_map['ready'][r.name] = False
                    init_shared_map(robots)
                    paused = False
                    start_time = None
                    step_count = 0
                    episode_steps = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not paused:
            # each robot senses and publishes its local scan first
            for r in robots:
                r.sense_and_publish()

            # wait until all robots have signaled ready at least once
            if not start_time:
                if all(shared_map['ready'].values()):
                    start_time = time.time()
            else:
                if time.time() - start_time >= countdown:
                    # let each robot take a learning step (order can be randomized)
                    order = list(range(len(robots)))
                    random.shuffle(order)
                    for idx in order:
                        robots[idx].step()
                        # persist shared map after each step to maximize knowledge sharing
                        save_shared_map(shared_map, robots)
                        episode_steps += 1

            step_count += 1

        # Draw current grid / shared knowledge
        screen.fill(WHITE)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if grid[y][x] == 1:
                    pygame.draw.rect(screen, BLACK, rect)
                elif (x,y) in shared_map['obstacles']:
                    pygame.draw.rect(screen, ORANGE, rect)
                elif (x,y) in shared_map['visited']:
                    pygame.draw.rect(screen, GRAY, rect)
        gx, gy = GOAL_POS
        pygame.draw.rect(screen, YELLOW, pygame.Rect(gx*CELL_SIZE, gy*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw robots
        for r in robots:
            rx, ry = r.pos
            pygame.draw.circle(screen, r.color, (rx*CELL_SIZE + CELL_SIZE//2, ry*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2)

        # optional: draw small text overlay with stats (epsilon of first agent)
        font = pygame.font.SysFont(None, 18)
        info = f"Robots: {NUM_ROBOTS} | Steps: {episode_steps} | Eps (R1): {robots[0].epsilon:.3f}"
        txt = font.render(info, True, (30,30,30))
        screen.blit(txt, (5,5))

        pygame.display.flip()

        # check termination: all reached goal OR all stuck OR exceeded steps
        all_reached = all(r.reached_goal for r in robots)
        # consider stuck if many robots flagged reached_goal artificially because no valid moves
        too_long = episode_steps > MAX_EPISODE_STEPS
        if all_reached or too_long:
            # write final shared map then reconstruct display
            save_shared_map(shared_map, robots)
            reconstruct_and_display(shared_map, robots, screen)
            return

    pygame.quit()

if __name__ == "__main__":
    main()
