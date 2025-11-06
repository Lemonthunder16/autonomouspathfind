# (Full modified script — paste over your existing file)
import pygame
import random
import time
import json
from collections import deque, defaultdict
import os
import heapq
import math

# =================== CONFIG ===================
GRID_SIZE = 50
CELL_SIZE = 12
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30
OBSTACLE_RATIO = 0.4
LIDAR_RADIUS = 5
MOVE_DELAY = 0.06
MAP_FILE = "shared_map.json"
STUCK_THRESHOLD = 8  # number of consecutive no-move ticks before fallback

# ------- Q-LEARNING TOGGLE & HYPERPARAMS -------
USE_QLEARNING = False      # <-- If False: NO behavior change. If True: agent used for fallback decisions.
Q_TABLE_FILE = "q_table.json"
Q_ALPHA = 0.5              # learning rate
Q_GAMMA = 0.9              # discount
Q_EPS_START = 0.2          # starting epsilon for exploration (epsilon-greedy)
Q_EPS_MIN = 0.02
Q_EPS_DECAY = 0.9995       # per-step decay

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
PINK = (255, 105, 180)
LIGHT_GREEN = (200, 255, 200)

START_POS = [
    (0, 0),
    (0, GRID_SIZE - 1),
    (GRID_SIZE - 1, 0),
    (GRID_SIZE - 1, GRID_SIZE - 1),
    (GRID_SIZE // 2, 0),
    (0, GRID_SIZE // 2)
]
GOAL_POS = (GRID_SIZE // 2, GRID_SIZE // 2)

# =================== UTILITIES ===================
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors(cell):
    x, y = cell
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            yield (nx, ny)

def write_shared_map_to_file(shared_map, robots):
    # Prepare JSON-friendly structure
    data = {
        "obstacles": sorted(list(shared_map["obstacles"])),
        "visited": sorted(list(shared_map["visited"])),
        "victory_path": list(shared_map["victory_path"]) if shared_map.get("victory_path") else [],
        "robots": {
            r.name: {
                "position": [r.pos[0], r.pos[1]],
                "reached_goal": r.reached_goal,
                "path_length": len(r.path)
            } for r in robots
        }
    }
    with open(MAP_FILE, "w") as f:
        json.dump(data, f, indent=2)

# =================== A* PATHFINDING ===================
def a_star(start, goal, grid, shared_obstacles):
    # Return a list of cells from start (excluded) to goal (included), or None if unreachable
    # Consider both actual grid obstacles and known shared obstacles.
    if start == goal:
        return [goal]

    def is_blocked(cell):
        x, y = cell
        return grid[y][x] == 1 or cell in shared_obstacles

    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))
    came_from = {}
    gscore = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        for nb in neighbors(current):
            if is_blocked(nb):
                continue
            tentative_g = gscore[current] + 1
            if tentative_g < gscore.get(nb, float('inf')):
                came_from[nb] = current
                gscore[nb] = tentative_g
                f = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_set, (f, tentative_g, nb))

    return None

# =================== GRID ===================
def generate_grid():
    while True:
        grid = [[0 if random.random() > OBSTACLE_RATIO else 1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        grid[GOAL_POS[1]][GOAL_POS[0]] = 0
        for sx, sy in START_POS:
            grid[sy][sx] = 0
        # ensure each start has some path to goal
        if all(path_exists(grid, (sx, sy), GOAL_POS) for sx, sy in START_POS):
            return grid

def path_exists(grid, start, goal):
    queue = deque([start])
    seen = set([start])
    while queue:
        c = queue.popleft()
        if c == goal:
            return True
        for nb in neighbors(c):
            x,y = nb
            if grid[y][x] == 0 and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return False

# =================== Q-LEARNING AGENT ===================
class QLearningAgent:
    """
    Q-table over compact local state:
    - 3x3 local occupancy around robot: each cell 0=open,1=blocked,2=unknown (but we use known grid for simplicity)
      flattened into a small integer (ternary encoding) OR we simplify to binary occupancy for size.
    - Relative direction to goal: discretized into 8 sectors (N,NE,E,SE,S,SW,W,NW) OR simple sign of dx/dy.
    Action: 0=left,1=right,2=up,3=down  (matching neighbor order we'll convert)
    """
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # left,right,up,down mapping

    def __init__(self, name="agent"):
        self.q = {}  # {(state): [q0,q1,q2,q3]}
        self.alpha = Q_ALPHA
        self.gamma = Q_GAMMA
        self.epsilon = Q_EPS_START
        self.name = name
        self.steps = 0

    def state_from(self, pos, grid, shared_obstacles, goal):
        # Build a compact state:
        # - occupancy of 3x3 centered on pos (0=open,1=blocked)
        # - relative direction: sign(dx), sign(dy) mapped to one of 9 values (-1,0,1 x -1,0,1)
        x0,y0 = pos
        occ = 0
        # binary encoding: 9 bits
        bit = 0
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                nx, ny = x0+dx, y0+dy
                blocked = 1
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if grid[ny][nx] == 0 and (nx,ny) not in shared_obstacles:
                        blocked = 0
                # set bit
                occ |= (blocked << bit)
                bit += 1
        dxg = goal[0] - x0
        dyg = goal[1] - y0
        sdx = 0 if dxg == 0 else (1 if dxg > 0 else -1)
        sdy = 0 if dyg == 0 else (1 if dyg > 0 else -1)
        dir_code = (sdx + 1) * 3 + (sdy + 1)  # 0..8
        # compress to tuple (occ, dir_code)
        return (occ, dir_code)

    def ensure_state(self, s):
        if s not in self.q:
            self.q[s] = [0.0 for _ in range(len(self.ACTIONS))]

    def choose_action(self, pos, grid, shared_obstacles, goal):
        s = self.state_from(pos, grid, shared_obstacles, goal)
        self.ensure_state(s)
        # epsilon-greedy
        if random.random() < self.epsilon:
            a = random.randrange(len(self.ACTIONS))
        else:
            qs = self.q[s]
            mx = max(qs)
            # tie breaking
            best = [i for i,qv in enumerate(qs) if qv == mx]
            a = random.choice(best)
        return a, s

    def learn(self, s, a, reward, s2):
        self.ensure_state(s)
        self.ensure_state(s2)
        qsa = self.q[s][a]
        q_next = max(self.q[s2]) if self.q[s2] else 0.0
        self.q[s][a] += self.alpha * (reward + self.gamma * q_next - qsa)
        # decay epsilon slowly
        self.steps += 1
        self.epsilon = max(Q_EPS_MIN, self.epsilon * Q_EPS_DECAY)

    def save(self, fname=Q_TABLE_FILE):
        # serialize q-table (convert keys to strings)
        out = {"alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon, "q": {}}
        for k,v in self.q.items():
            keystr = f"{k[0]}|{k[1]}"
            out["q"][keystr] = v
        with open(fname, "w") as f:
            json.dump(out, f)

    def load(self, fname=Q_TABLE_FILE):
        if not os.path.exists(fname):
            return False
        with open(fname, "r") as f:
            data = json.load(f)
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.q = {}
        for keystr, v in data.get("q", {}).items():
            occ_str, dir_str = keystr.split("|")
            s = (int(occ_str), int(dir_str))
            self.q[s] = v
        return True

# =================== ROBOT ===================
class Robot:
    def __init__(self, name, color, start, grid, shared_map, qagent=None):
        self.name = name
        self.color = color
        self.pos = start
        self.grid = grid
        self.shared_map = shared_map
        self.reached_goal = False
        self.last_move_time = 0
        self.visited = set()
        self.frontier = deque()
        self.path = [start]  # historical path
        self.planned = []     # current planned path to follow (list of cells)
        self.following_victory = False
        self.victory_target = None
        self.stuck_counter = 0
        self.prev_pos = start
        self.qagent = qagent  # optional Q-learning agent
        # For Q-learning transitions
        self._last_state = None
        self._last_action = None

    def scan_and_update(self):
        # Scan around and add obstacles to shared map
        x0, y0 = self.pos
        seen_obstacles = set()
        for dx in range(-LIDAR_RADIUS, LIDAR_RADIUS+1):
            for dy in range(-LIDAR_RADIUS, LIDAR_RADIUS+1):
                nx, ny = x0+dx, y0+dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if self.grid[ny][nx] == 1:
                        seen_obstacles.add((nx, ny))
        # Merge into shared map
        before = len(self.shared_map['obstacles'])
        self.shared_map['obstacles'].update(seen_obstacles)
        # Mark current as visited
        self.shared_map['visited'].add(self.pos)
        # Indicate ready
        self.shared_map['ready'][self.name] = True

    def fallback_escape(self):
        # Try random moves around (prefer unvisited, then any valid)
        # If Q-learning enabled and agent present, use it to pick best fallback action
        opts = []
        for nb in neighbors(self.pos):
            x,y = nb
            if self.grid[y][x] == 0 and nb not in self.shared_map['obstacles']:
                opts.append(nb)
        # if Q learning is active, consult agent
        if USE_QLEARNING and self.qagent:
            # choose action using agent, but validate move
            a, s = self.qagent.choose_action(self.pos, self.grid, self.shared_map['obstacles'], GOAL_POS)
            dx,dy = QLearningAgent.ACTIONS[a]
            cand = (self.pos[0] + dx, self.pos[1] + dy)
            # ensure candidate is on-grid and not blocked
            valid = (0 <= cand[0] < GRID_SIZE and 0 <= cand[1] < GRID_SIZE and
                     self.grid[cand[1]][cand[0]] == 0 and cand not in self.shared_map['obstacles'])
            if not valid:
                # try to pick another valid action greedily from Q-values
                self.qagent.ensure_state(s)
                qvals = list(enumerate(self.qagent.q[s]))
                # sort by q desc
                qvals.sort(key=lambda x: x[1], reverse=True)
                selected = None
                for act, _ in qvals:
                    dx2,dy2 = QLearningAgent.ACTIONS[act]
                    cand2 = (self.pos[0] + dx2, self.pos[1] + dy2)
                    if 0 <= cand2[0] < GRID_SIZE and 0 <= cand2[1] < GRID_SIZE and \
                       self.grid[cand2[1]][cand2[0]] == 0 and cand2 not in self.shared_map['obstacles']:
                        selected = (act, cand2)
                        break
                if selected:
                    a, cand = selected
                    valid = True
            # If still invalid, fall back to normal behavior
            if valid:
                # record last state/action for learning after the move
                self._last_state = s
                self._last_action = a
                self._do_move(cand)
                # compute immediate reward and next state to learn
                # reward: closer to goal +1, reach goal +50, step penalty -0.1, invalid move large negative (handled above)
                r = -0.1
                if self.pos == GOAL_POS:
                    r += 50.0
                else:
                    # distance delta
                    old_dist = manhattan((self.pos[0]-dx, self.pos[1]-dy), GOAL_POS) if False else None
                    # easier: compare previous position stored in prev_pos
                    prevd = manhattan(self.prev_pos, GOAL_POS)
                    nowd = manhattan(self.pos, GOAL_POS)
                    if nowd < prevd:
                        r += 1.0
                    elif nowd > prevd:
                        r -= 0.5
                # next state
                s2 = self.qagent.state_from(self.pos, self.grid, self.shared_map['obstacles'], GOAL_POS)
                self.qagent.learn(self._last_state, self._last_action, r, s2)
                return True

        # prefer not-visited
        random.shuffle(opts)
        best = None
        for o in opts:
            if o not in self.shared_map['visited']:
                best = o
                break
        if not best and opts:
            best = opts[0]
        if best:
            self._do_move(best)
            return True
        return False

    def _do_move(self, newpos):
        if newpos == self.pos:
            return
        # store prev for reward calc
        self.prev_pos = self.pos
        self.pos = newpos
        self.path.append(newpos)
        self.shared_map['visited'].add(newpos)
        # reset stuck state if actually moved
        if newpos != self.prev_pos:
            self.stuck_counter = 0
            self.prev_pos = newpos
        else:
            self.stuck_counter += 1

    def plan_to_point(self, target):
        # Use A* to plan path to target. path returned includes target and excludes current pos.
        plan = a_star(self.pos, target, self.grid, self.shared_map['obstacles'])
        if plan:
            self.planned = plan
            return True
        return False

    def move_along_planned(self):
        if not self.planned:
            return False
        nextcell = self.planned.pop(0)
        # If blocked unexpectedly, abort plan
        if nextcell in self.shared_map['obstacles'] or self.grid[nextcell[1]][nextcell[0]] == 1:
            self.planned = []
            return False
        self._do_move(nextcell)
        return True

    def move_step(self):
        if self.reached_goal:
            return

        if time.time() - self.last_move_time < MOVE_DELAY:
            return
        self.last_move_time = time.time()

        # Update scan info already done externally, but safe to call
        # self.scan_and_update()

        # If victory path exists, target nearest point then follow
        vpath = self.shared_map.get('victory_path')
        if vpath:
            # If we're not currently following victory sequence, prepare
            if not self.victory_target:
                # Choose nearest reachable point on victory path
                best = None
                best_dist = float('inf')
                for p in vpath:
                    d = manhattan(self.pos, p)
                    if d < best_dist:
                        # only consider if there's some path (A*) to it
                        plan = a_star(self.pos, p, self.grid, self.shared_map['obstacles'])
                        if plan:
                            best_dist = d
                            best = p
                if best is None:
                    # No path to any victory cell (rare). Fallback: try to move greedily closer.
                    # Use greedy single-step toward the nearest victory cell
                    nearest = min(vpath, key=lambda c: manhattan(self.pos, c))
                    self.move_toward_greedy(nearest)
                    return
                self.victory_target = best
                # plan path to that target
                self.plan_to_point(self.victory_target)

            # If we have a planned route to the victory target, follow it
            if self.planned:
                moved = self.move_along_planned()
                if not moved:
                    # abort and try fallback
                    self.planned = []
                    self.stuck_counter += 1
            else:
                # If no plan exists (maybe we're at the victory_target), start following the shared victory path from that point
                if self.pos == self.victory_target:
                    # follow remaining segment on vpath
                    idx = None
                    for i, c in enumerate(vpath):
                        if c == self.pos:
                            idx = i
                            break
                    if idx is None:
                        # safety: reset target
                        self.victory_target = None
                        return
                    # set following_victory and set planned to remaining vpath (including next cell)
                    self.following_victory = True
                    self.planned = list(vpath[idx+1:])  # next steps toward goal
                    if not self.planned:
                        # we are at goal or last point
                        if self.pos == GOAL_POS:
                            self.reached_goal = True
                        return
                    moved = self.move_along_planned()
                    if not moved:
                        self.planned = []
                        self.stuck_counter += 1
                else:
                    # try to plan again
                    if not self.plan_to_point(self.victory_target):
                        # if can't, reset and fallback
                        self.victory_target = None

            # if somehow reached goal during following
            if self.pos == GOAL_POS:
                self.reached_goal = True
                # ensure shared victory set
                if not self.shared_map.get('victory_path'):
                    self.shared_map['victory_path'] = list(self.path)
            return

        # ===== Normal exploration (no victory path yet) =====
        # Add neighbors to frontier
        x,y = self.pos
        self.visited.add((x,y))
        self.shared_map['visited'].add((x,y))

        for nb in neighbors(self.pos):
            if nb not in self.visited and nb not in self.shared_map['obstacles'] and self.grid[nb[1]][nb[0]] == 0:
                if nb not in self.frontier:
                    self.frontier.append(nb)

        # Greedy pick neighbor closer to goal from frontier
        next_cell = None
        sorted_neighbors = sorted(self.frontier, key=lambda c: manhattan(c, GOAL_POS))
        for cell in sorted_neighbors:
            if cell not in self.shared_map['obstacles'] and cell not in self.visited and self.grid[cell[1]][cell[0]] == 0:
                next_cell = cell
                break

        if next_cell:
            self.frontier.remove(next_cell)
            self._do_move(next_cell)
        elif self.frontier:
            # fallback to any in frontier
            cand = self.frontier.popleft()
            self._do_move(cand)
        else:
            # no frontier: try fallback escape
            escaped = self.fallback_escape()
            if not escaped:
                # nowhere to go; increment stuck counter
                self.stuck_counter += 1

        # If stuck for too long attempt random valid move
        if self.stuck_counter >= STUCK_THRESHOLD:
            if self.fallback_escape():
                self.stuck_counter = 0
            else:
                # try random neighbor ignoring visited but not obstacles
                opts = [nb for nb in neighbors(self.pos) if self.grid[nb[1]][nb[0]] == 0 and nb not in self.shared_map['obstacles']]
                if opts:
                    self._do_move(random.choice(opts))
                    self.stuck_counter = 0

        # Victory detection
        if self.pos == GOAL_POS:
            self.reached_goal = True
            # If not set, publish our path as victory path
            if not self.shared_map.get('victory_path'):
                self.shared_map['victory_path'] = list(self.path)

    def move_toward_greedy(self, target):
        # single-step greedy toward target (used as last resort)
        x,y = self.pos
        tx,ty = target
        options = []
        for nb in neighbors(self.pos):
            nx,ny = nb
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[ny][nx] == 0 and nb not in self.shared_map['obstacles']:
                options.append(nb)
        if not options:
            return False
        nextcell = min(options, key=lambda c: manhattan(c, target))
        self._do_move(nextcell)
        return True

# =================== SHARED MAP INIT ===================
def init_shared_map(robots):
    shared_map = {
        "obstacles": set(),
        "visited": set(),
        "victory_path": [],
        "ready": {r.name: False for r in robots}
    }
    # write initial file
    write_shared_map_to_file(shared_map, robots)
    return shared_map

# =================== MAIN ===================
def main():
    global USE_QLEARNING
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Swarm Robots Exploration - Robust Victory Tracing + Q-Learning (toggleable)")
    clock = pygame.time.Clock()

    grid = generate_grid()
    # create a shared Q-agent for all robots (optional)
    qagent = QLearningAgent(name="shared_agent")
    # Don't auto-load; allow user to press 'L' to load

    robots = [
        Robot("R1", RED, START_POS[0], grid, None, qagent),
        Robot("R2", BLUE, START_POS[1], grid, None, qagent),
        Robot("R3", GREEN, START_POS[2], grid, None, qagent),
        Robot("R4", PURPLE, START_POS[3], grid, None, qagent),
        Robot("R5", CYAN, START_POS[4], grid, None, qagent),
        Robot("R6", PINK, START_POS[5], grid, None, qagent)
    ]
    shared_map = {
        "obstacles": set(),
        "visited": set(),
        "victory_path": [],
        "ready": {r.name: False for r in robots}
    }
    # attach shared_map to robots
    for r in robots:
        r.shared_map = shared_map

    init_shared_map(robots)  # also writes file

    paused = False
    countdown = 2.0
    start_time = None

    running = True
    print("Simulation started. Q-learning is", "ENABLED" if USE_QLEARNING else "DISABLED")
    print("Key controls: SPACE pause, R reset, ESC quit, Q toggle Q-learning, S save Q-table, L load Q-table")

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # reset everything
                    grid = generate_grid()
                    robots = [
                        Robot("R1", RED, START_POS[0], grid, None, qagent),
                        Robot("R2", BLUE, START_POS[1], grid, None, qagent),
                        Robot("R3", GREEN, START_POS[2], grid, None, qagent),
                        Robot("R4", PURPLE, START_POS[3], grid, None, qagent),
                        Robot("R5", CYAN, START_POS[4], grid, None, qagent),
                        Robot("R6", PINK, START_POS[5], grid, None, qagent)
                    ]
                    shared_map = {
                        "obstacles": set(),
                        "visited": set(),
                        "victory_path": [],
                        "ready": {r.name: False for r in robots}
                    }
                    for r in robots:
                        r.shared_map = shared_map
                    init_shared_map(robots)
                    paused = False
                    start_time = None
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_q:
                    USE_QLEARNING = not USE_QLEARNING
                    print("Toggled Q-learning:", "ENABLED" if USE_QLEARNING else "DISABLED")
                elif event.key == pygame.K_s:
                    qagent.save()
                    print("Saved Q-table to", Q_TABLE_FILE)
                elif event.key == pygame.K_l:
                    ok = qagent.load()
                    print("Loaded Q-table from", Q_TABLE_FILE, ok)

        if not paused:
            # Each robot scans and updates shared obstacles/visited
            for r in robots:
                r.scan_and_update()

            # When all ready, start moves after countdown
            if not start_time:
                if all(shared_map['ready'].values()):
                    start_time = time.time()
            else:
                if time.time() - start_time >= countdown:
                    for r in robots:
                        r.move_step()

            # write shared map to JSON file
            write_shared_map_to_file(shared_map, robots)

        # Draw grid
        screen.fill(WHITE)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if grid[y][x] == 1:
                    pygame.draw.rect(screen, BLACK, rect)
                elif (x,y) in shared_map['obstacles']:
                    pygame.draw.rect(screen, ORANGE, rect)
                elif (x,y) in shared_map['visited']:
                    pygame.draw.rect(screen, (220,220,220), rect)

        # draw victorious path
        if shared_map.get('victory_path'):
            for (vx, vy) in shared_map['victory_path']:
                pygame.draw.rect(screen, LIGHT_GREEN, pygame.Rect(vx*CELL_SIZE, vy*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw goal
        gx, gy = GOAL_POS
        pygame.draw.rect(screen, YELLOW, pygame.Rect(gx*CELL_SIZE, gy*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw robots and their trails (individual colored dots)
        for r in robots:
            # draw trail (small squares)
            for p in r.path[-150:]:  # only draw last 150 for readability
                rect = pygame.Rect(p[0]*CELL_SIZE + CELL_SIZE//4, p[1]*CELL_SIZE + CELL_SIZE//4, CELL_SIZE//2, CELL_SIZE//2)
                pygame.draw.rect(screen, r.color, rect)
            # draw current pos as circle
            rx, ry = r.pos
            pygame.draw.circle(screen, r.color, (rx*CELL_SIZE + CELL_SIZE//2, ry*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2)

        pygame.display.flip()

        # Pause simulation if everyone reached goal
        if all(r.reached_goal for r in robots):
            paused = True

    pygame.quit()

if __name__ == "__main__":
    main()
