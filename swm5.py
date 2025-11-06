import pygame
import random
import time
import json
from collections import deque, defaultdict
import os
import heapq

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

# =================== SAFE, CONSERVATIVE Q-LEARNING (TIE-BREAK ONLY) ===================
class SafeQL:
    """
    Conservative Q-learning that *lightly* influences choices:
    - Only tie-breaks among equally-good candidates (same Manhattan distance).
    - Activates after warm-up steps.
    - Auto-disables if drift > 5% vs baseline path length.
    """
    # Master toggles
    enabled = True
    apply_q_policy = True        # allows Q tie-breaks (not full control)
    td_clip = 1e-3

    # Hyperparams (very conservative)
    alpha = 2e-4                 # tiny learning rate
    gamma = 0.99
    epsilon = 0.0                # no global exploration

    # Influence policy
    warmup_steps = 500           # learn quietly first
    q_influence_prob = 0.25      # only 25% of tie-breaks use Q
    epsilon_if_stuck = 0.02      # tiny exploration only when stuck
    drift_cap = 0.05             # 5% allowed deterioration
    drift_guard_enabled = True

    # Action mapping (4-connected grid)
    DIR_TO_A = {(-1,0):0, (1,0):1, (0,-1):2, (0,1):3}
    A_TO_DIR = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
    ACTIONS = [0,1,2,3]

    def __init__(self, agent_id="agent"):
        self.agent_id = agent_id
        self.Q = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # Q[(x,y)][a]
        self.steps = 0

    @staticmethod
    def encode_state(pos):
        # State = position only (keeps stability & speed)
        return (int(pos[0]), int(pos[1]))

    def observe(self, s, a, r, s2, done=False):
        """Passive Q-learning update from transition (s,a,r,s2)."""
        if not SafeQL.enabled or a is None:
            return
        qsa = self.Q[s][a]
        max_next = 0.0 if done else max(self.Q[s2])
        target = r + SafeQL.gamma * max_next
        td = target - qsa
        # clamp updates to avoid drift
        if td > SafeQL.td_clip: td = SafeQL.td_clip
        elif td < -SafeQL.td_clip: td = -SafeQL.td_clip
        self.Q[s][a] = qsa + SafeQL.alpha * td
        self.steps += 1

    def should_use_q(self):
        if not SafeQL.apply_q_policy or not SafeQL.enabled:
            return False
        if self.steps < SafeQL.warmup_steps:
            return False
        return random.random() < SafeQL.q_influence_prob

    def pick_best_among(self, s, candidates, stuck_counter):
        """Choose among equally-good candidate cells using Q (tie-break only)."""
        if not candidates:
            return None
        # tiny 'epsilon if stuck': only when struggling, allow a nudge
        if stuck_counter >= STUCK_THRESHOLD // 2 and random.random() < SafeQL.epsilon_if_stuck:
            return random.choice(candidates)
        if not self.should_use_q():
            return random.choice(candidates)  # fair random among equals
        best, best_q = None, -1e18
        for c in candidates:
            dx, dy = c[0]-s[0], c[1]-s[1]
            a = SafeQL.DIR_TO_A.get((dx,dy), None)
            q = self.Q[s][a] if a is not None else -1e9
            if q > best_q:
                best_q, best = q, c
        return best

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

# =================== ROBOT ===================
class Robot:
    def __init__(self, name, color, start, grid, shared_map):
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

        # Q-learning (conservative) instance
        self.ql = SafeQL(agent_id=self.name)

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
        self.shared_map['obstacles'].update(seen_obstacles)
        # Mark current as visited
        self.shared_map['visited'].add(self.pos)
        # Indicate ready
        self.shared_map['ready'][self.name] = True

    def fallback_escape(self):
        # Try random moves around (prefer unvisited, then any valid)
        opts = []
        for nb in neighbors(self.pos):
            x,y = nb
            if self.grid[y][x] == 0 and nb not in self.shared_map['obstacles']:
                opts.append(nb)
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
        # ---- Passive Q-learning update from actual move ----
        s = SafeQL.encode_state(self.pos)
        dx, dy = newpos[0]-self.pos[0], newpos[1]-self.pos[1]
        a = SafeQL.DIR_TO_A.get((dx, dy), None)
        done = (newpos == GOAL_POS)
        # tiny, shaping reward: -0.001 per step, +1.0 on reaching goal
        r = (1.0 if done else -0.001)

        # Perform the move (no control change here)
        self.pos = newpos
        self.path.append(newpos)
        self.shared_map['visited'].add(newpos)

        # reset stuck state if actually moved
        if newpos != self.prev_pos:
            self.stuck_counter = 0
            self.prev_pos = newpos
        else:
            self.stuck_counter += 1

        # Q-learning observe
        s2 = SafeQL.encode_state(self.pos)
        self.ql.observe(s, a, r, s2, done=done)

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

        # If victory path exists, target nearest point then follow
        vpath = self.shared_map.get('victory_path')
        if vpath:
            if not self.victory_target:
                best = None
                best_dist = float('inf')
                for p in vpath:
                    d = manhattan(self.pos, p)
                    if d < best_dist:
                        plan = a_star(self.pos, p, self.grid, self.shared_map['obstacles'])
                        if plan:
                            best_dist = d
                            best = p
                if best is None:
                    nearest = min(vpath, key=lambda c: manhattan(self.pos, c))
                    self.move_toward_greedy(nearest)
                    return
                self.victory_target = best
                self.plan_to_point(self.victory_target)

            if self.planned:
                moved = self.move_along_planned()
                if not moved:
                    self.planned = []
                    self.stuck_counter += 1
            else:
                if self.pos == self.victory_target:
                    idx = None
                    for i, c in enumerate(vpath):
                        if c == self.pos:
                            idx = i
                            break
                    if idx is None:
                        self.victory_target = None
                        return
                    self.following_victory = True
                    self.planned = list(vpath[idx+1:])
                    if not self.planned:
                        if self.pos == GOAL_POS:
                            self.reached_goal = True
                        return
                    moved = self.move_along_planned()
                    if not moved:
                        self.planned = []
                        self.stuck_counter += 1
                else:
                    if not self.plan_to_point(self.victory_target):
                        self.victory_target = None

            if self.pos == GOAL_POS:
                self.reached_goal = True
                if not self.shared_map.get('victory_path'):
                    self.shared_map['victory_path'] = list(self.path)
            return

        # ===== Normal exploration (no victory path yet) =====
        x,y = self.pos
        self.visited.add((x,y))
        self.shared_map['visited'].add((x,y))

        for nb in neighbors(self.pos):
            if nb not in self.visited and nb not in self.shared_map['obstacles'] and self.grid[nb[1]][nb[0]] == 0:
                if nb not in self.frontier:
                    self.frontier.append(nb)

        # --- Tie-break among equally good neighbors using Q (very light influence) ---
        next_cell = None
        if self.frontier:
            # rank frontier by Manhattan distance (your original heuristic)
            by_d = {}
            for c in self.frontier:
                if c in self.shared_map['obstacles'] or self.grid[c[1]][c[0]] == 1:
                    continue
                d = manhattan(c, GOAL_POS)
                by_d.setdefault(d, []).append(c)

            if by_d:
                best_d = min(by_d.keys())
                best_bucket = by_d[best_d]

                if len(best_bucket) > 1:
                    # Q only breaks ties after warm-up; tiny explore only when getting stuck
                    next_cell = self.ql.pick_best_among(self.pos, best_bucket, self.stuck_counter)
                else:
                    next_cell = best_bucket[0]

        if next_cell:
            self.frontier.remove(next_cell)
            self._do_move(next_cell)
        elif self.frontier:
            cand = self.frontier.popleft()
            self._do_move(cand)
        else:
            escaped = self.fallback_escape()
            if not escaped:
                self.stuck_counter += 1

        # If stuck for too long attempt random valid move
        if self.stuck_counter >= STUCK_THRESHOLD:
            if self.fallback_escape():
                self.stuck_counter = 0
            else:
                opts = [nb for nb in neighbors(self.pos) if self.grid[nb[1]][nb[0]] == 0 and nb not in self.shared_map['obstacles']]
                if opts:
                    self._do_move(random.choice(opts))
                    self.stuck_counter = 0

        # Victory detection
        if self.pos == GOAL_POS:
            self.reached_goal = True
            if not self.shared_map.get('victory_path'):
                self.shared_map['victory_path'] = list(self.path)

    def move_toward_greedy(self, target):
        # single-step greedy toward target (used as last resort)
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
    write_shared_map_to_file(shared_map, robots)
    return shared_map

# =================== MAIN ===================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Swarm Robots Exploration - Robust Victory Tracing (with safe Q-learning tie-breaks)")
    clock = pygame.time.Clock()

    grid = generate_grid()
    robots = [
        Robot("R1", RED, START_POS[0], grid, None),
        Robot("R2", BLUE, START_POS[1], grid, None),
        Robot("R3", GREEN, START_POS[2], grid, None),
        Robot("R4", PURPLE, START_POS[3], grid, None),
        Robot("R5", CYAN, START_POS[4], grid, None),
        Robot("R6", PINK, START_POS[5], grid, None)
    ]
    shared_map = {
        "obstacles": set(),
        "visited": set(),
        "victory_path": [],
        "ready": {r.name: False for r in robots}
    }
    for r in robots:
        r.shared_map = shared_map

    init_shared_map(robots)  # also writes file

    paused = False
    countdown = 2.0
    start_time = None

    running = True
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
                        Robot("R1", RED, START_POS[0], grid, None),
                        Robot("R2", BLUE, START_POS[1], grid, None),
                        Robot("R3", GREEN, START_POS[2], grid, None),
                        Robot("R4", PURPLE, START_POS[3], grid, None),
                        Robot("R5", CYAN, START_POS[4], grid, None),
                        Robot("R6", PINK, START_POS[5], grid, None)
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
                    # Re-enable Q for a fresh run
                    SafeQL.apply_q_policy = True
                elif event.key == pygame.K_ESCAPE:
                    running = False

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

            # set baseline once: first victory path defines baseline
            if shared_map.get('victory_path') and 'baseline_len' not in shared_map:
                shared_map['baseline_len'] = len(shared_map['victory_path'])
                shared_map['runs'] = 0
                shared_map['avg_len'] = float(len(shared_map['victory_path']))

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
            # draw current pos as circle (centered)
            rx, ry = r.pos
            pygame.draw.circle(screen, r.color, (rx*CELL_SIZE + CELL_SIZE//2, ry*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2)

        pygame.display.flip()

        # Pause simulation if everyone reached goal; update drift guard stats
        if all(r.reached_goal for r in robots):
            # compute mean path length this run
            run_len = sum(len(r.path) for r in robots) / max(1, len(robots))
            if 'runs' not in shared_map:
                shared_map['runs'] = 0
                shared_map['avg_len'] = run_len
            shared_map['runs'] += 1
            shared_map['avg_len'] = 0.9 * shared_map['avg_len'] + 0.1 * run_len

            # drift guard: disable Q if average path length > baseline * (1 + 5%)
            if 'baseline_len' in shared_map and SafeQL.drift_guard_enabled:
                baseline = shared_map['baseline_len']
                if baseline > 0 and shared_map['avg_len'] > (1.0 + SafeQL.drift_cap) * baseline:
                    SafeQL.apply_q_policy = False  # auto-disable Q influence

            paused = True

    pygame.quit()

if __name__ == "__main__":
    main()
