# controllers/swarm_onefile/swarm_onefile.py
# Swarm A*: single file, shared JSON "memory", synchronized start, safe-distance inflation,
# and 3-decimal logging of every new obstacle point (world x,y) to shared_map.json.

from controller import Robot
import math, json, os, time, random, heapq

# ====== WORLD / GRID ======
RES = 0.05                           # 5 cm grid
X_MIN, X_MAX = -2.0, 2.0
Y_MIN, Y_MAX = -2.0, 2.0
W = int((X_MAX - X_MIN) / RES)
H = int((Y_MAX - Y_MIN) / RES)
GOAL_WORLD = (0.0, 0.0)              # go to center

# ====== ROBOT / CONTROL ======
DT = 64
LIDAR_MAX = 1.5
FRONT_STOP = 0.15                    # stop if something is really close
LOOKAHEAD = 0.20
BASE_SPEED = 4.2
MAX_WHEEL  = 6.28

# Safe distance (inflate obstacles)
EPUCK_RADIUS_M   = 0.035
SAFE_MARGIN_M    = 0.065             # tweak this for more/less clearance
INFLATE_R_CELLS  = max(1, int(round((EPUCK_RADIUS_M + SAFE_MARGIN_M) / RES)))

# ====== SYNC (start together) ======
MIN_SWARM = 2                         # number of robots you expect
START_DELAY_S = 1.0                   # count-down once everyone is ready

# ====== SHARED FILES ======
HERE = os.path.dirname(__file__)
SHARED_FILE = os.path.join(HERE, "shared_map.json")
LOCK_FILE   = os.path.join(HERE, "shared_map.lock")

# ---------- small utils ----------
def clamp(x,a,b): return max(a, min(b, x))
def wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a
def r3(v):  # round to 3 decimals as float
    return float(f"{v:.3f}")

def world_to_grid(x, y):
    gx = int((x - X_MIN) / RES); gy = int((y - Y_MIN) / RES)
    return max(0,min(W-1,gx)), max(0,min(H-1,gy))
def grid_to_world(gx, gy):
    return X_MIN + (gx + 0.5)*RES, Y_MIN + (gy + 0.5)*RES
def neighbors8(gx, gy):
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx==0 and dy==0: continue
            yield gx+dx, gy+dy

def a_star(occ, s, g):
    sx, sy = s; gx, gy = g
    if occ[sx][sy] or occ[gx][gy]: return []
    best = {(sx,sy):0.0}; parent = {}
    pq = [(abs(gx-sx)+abs(gy-sy), 0.0, (sx,sy), None)]
    def h(x,y): return abs(x-gx)+abs(y-gy)
    while pq:
        f, gc, node, prev = heapq.heappop(pq)
        if node in parent: continue
        parent[node] = prev
        if node == (gx,gy):
            path = [node]
            while parent[path[-1]] is not None: path.append(parent[path[-1]])
            return list(reversed(path))
        x,y = node
        for nx, ny in neighbors8(x,y):
            if not (0<=nx<W and 0<=ny<H): continue
            if occ[nx][ny]: continue
            step = 1.4142 if (nx!=x and ny!=y) else 1.0
            ng = gc + step
            if best.get((nx,ny), 1e9) <= ng: continue
            best[(nx,ny)] = ng
            heapq.heappush(pq, (ng + h(nx,ny), ng, (nx,ny), node))
    return []

# ---------- shared JSON helpers (world x,y decimals) ----------
def ensure_shared_file():
    # New schema: {"blocked_xy": [[x,y],...], "ready": {...}, "epoch": ...}
    if not os.path.exists(SHARED_FILE):
        with open(SHARED_FILE, "w", encoding="utf-8") as f:
            json.dump({"blocked_xy": [], "ready": {}, "epoch": None}, f)

def read_shared():
    try:
        with open(SHARED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"blocked_xy": [], "ready": {}, "epoch": None}

def with_lock_update(update_fn):
    start=time.time()
    while os.path.exists(LOCK_FILE):
        if time.time()-start>1.5: return
        time.sleep(0.01)
    try:
        open(LOCK_FILE,"x").close()
        data = read_shared()
        changed = update_fn(data) or False
        if changed:
            with open(SHARED_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
    finally:
        try: os.remove(LOCK_FILE)
        except: pass

def mark_occ_inflated(occ, gx, gy, r_cells):
    r2 = r_cells*r_cells
    for dx in range(-r_cells, r_cells+1):
        for dy in range(-r_cells, r_cells+1):
            if dx*dx + dy*dy <= r2:
                nx, ny = gx+dx, gy+dy
                if 0<=nx<W and 0<=ny<H:
                    occ[nx][ny] = 1

# Local inflation helper for raw lidar grid cells (kept from your working version)
def inflate_cells(cells, r_cells):
    out = set()
    r2 = r_cells * r_cells
    for (gx, gy) in cells:
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                if dx*dx + dy*dy <= r2:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        out.add((nx, ny))
    return out

# ---------- Controller ----------
class SwarmOneFile:
    def __init__(self):
        self.r = Robot()
        self.name = self.r.getName()
        print(f"[{self.name}] START; shared='{SHARED_FILE}'  inflate={INFLATE_R_CELLS} cells")

        # motors
        self.lm = self.r.getDevice('left wheel motor')
        self.rm = self.r.getDevice('right wheel motor')
        self.lm.setPosition(float('inf')); self.rm.setPosition(float('inf'))
        self.lm.setVelocity(0.0); self.rm.setVelocity(0.0)

        # devices (same as your working build)
        self.gps = self.r.getDevice('gps');      self.gps.enable(DT)
        self.com = self.r.getDevice('compass');  self.com.enable(DT)
        self.lid = self.r.getDevice('lidar');    self.lid.enable(DT)

        ensure_shared_file()
        self.occ = [[0]*H for _ in range(W)]
        self.goal_g = world_to_grid(*GOAL_WORLD)
        self.path = []; self.idx = 0
        self.local_hits = set()
        self.last_merge = 0.0
        self.last_flush = 0.0
        self.started = False

        # small stagger so I/O doesn't collide
        for _ in range(int(random.uniform(0,5))):
            if self.r.step(DT) == -1: return

        self.barrier_sync()   # wait until both robots are ready

    # ---- sync barrier: all start together ----
    def barrier_sync(self):
        def add_ready(d):
            d.setdefault("ready", {})[self.name] = 1
            if d.get("epoch") is None and len(d["ready"]) >= MIN_SWARM:
                d["epoch"] = self.r.getTime() + START_DELAY_S
            return True
        with_lock_update(add_ready)

        while True:
            data = read_shared()
            epoch = data.get("epoch", None)
            if epoch is not None and self.r.getTime() >= epoch:
                print(f"[{self.name}] STARTING at t={self.r.getTime():.2f} (epoch={epoch:.2f})")
                self.started = True
                return
            self.lm.setVelocity(0.0); self.rm.setVelocity(0.0)
            if self.r.step(DT) == -1: return

    # ---- helpers ----
    def pose(self):
        x,y,_ = self.gps.getValues()
        cx,cy,_ = self.com.getValues()
        yaw = math.atan2(cx, cy)
        return x,y,yaw

    def merge_shared_into_occ(self):
        # Rebuild occupancy from shared world points (and honor legacy 'blocked' cells)
        self.occ = [[0]*H for _ in range(W)]
        data = read_shared()

        # New format: blocked_xy -> (x,y) world meters (3 decimals)
        for pt in data.get("blocked_xy", []):
            try:
                wx, wy = float(pt[0]), float(pt[1])
                gx, gy = world_to_grid(wx, wy)
                mark_occ_inflated(self.occ, gx, gy, INFLATE_R_CELLS)
            except Exception:
                continue

        # Back-compat: legacy grid entries if present
        for g in data.get("blocked", []):
            try:
                gx, gy = int(g[0]), int(g[1])
                mark_occ_inflated(self.occ, gx, gy, INFLATE_R_CELLS)
            except Exception:
                continue

    def lidar_cells(self):
        # Convert lidar returns -> grid cells
        rng = self.lid.getRangeImage()
        n = len(rng)
        if n < 3: return set()
        try: fov = self.lid.getFov()
        except: fov = math.pi
        x,y,yaw = self.pose()
        cells = set()
        for i, r in enumerate(rng):
            if not (0.06 <= r < (LIDAR_MAX - 1e-3)):  # ignore zeros/inf
                continue
            a_local = -0.5*fov + (i/(n-1))*fov
            ang = yaw + a_local
            hx = x + r*math.cos(ang)
            hy = y + r*math.sin(ang)
            cells.add(world_to_grid(hx, hy))
        return cells

    def save_world_points(self, pts_world):
        """Write world (x,y) points to shared 'blocked_xy' (3 decimals) and print log."""
        if not pts_world: return
        pts = [(r3(x), r3(y)) for (x,y) in pts_world]
        print(f"[{self.name}] NEW_OBS: " + ", ".join([f"({x:.3f},{y:.3f})" for (x,y) in pts]))
        def updater(d):
            arr = d.setdefault("blocked_xy", [])
            have = {(r3(p[0]), r3(p[1])) for p in arr}
            changed = False
            for (wx,wy) in pts:
                if (wx,wy) not in have:
                    arr.append([wx,wy]); changed = True
            return changed
        with_lock_update(updater)

    def save_new_cells(self, new_cells):
        # INPUT: grid cells. Convert to world and delegate to save_world_points
        if not new_cells: return
        world_pts = [grid_to_world(gx,gy) for (gx,gy) in sorted(new_cells)]
        self.save_world_points(world_pts)

    def plan_if_needed(self):
        x,y,_ = self.pose()
        s = world_to_grid(x,y)
        need = (not self.path) or (self.idx >= len(self.path))
        if not need:
            for (gx,gy) in self.path[self.idx:]:
                if self.occ[gx][gy]: need = True; break
        if not need: return
        self.merge_shared_into_occ()
        self.path = a_star(self.occ, s, self.goal_g)
        self.idx = 0
        print(f"[{self.name}] planned len={len(self.path)} from {s} -> {self.goal_g}")

    # ---- main step ----
    def step(self):
        now = self.r.getTime()

        # Perception -> shared memory
        raw = self.lidar_cells()                          # set of grid cells
        inflated = inflate_cells(raw, INFLATE_R_CELLS)    # inflate locally
        self.local_hits |= inflated

        # flush every 0.4 s (write as world decimals to blocked_xy)
        if now - self.last_flush > 0.4 and self.local_hits:
            self.save_new_cells(self.local_hits.copy())
            self.local_hits.clear()
            self.last_flush = now

        # rebuild occ from shared periodically
        if now - self.last_merge > 0.4:
            self.merge_shared_into_occ()
            self.last_merge = now

        self.plan_if_needed()

        # stop only if truly blocked very close
        rng = self.lid.getRangeImage()
        n = len(rng); mid = n//2
        front = [v for v in rng[max(0,mid-15):min(n,mid+15)] if 0.06 < v < (LIDAR_MAX-1e-3)]
        if front and min(front) < FRONT_STOP:
            self.lm.setVelocity(0.6); self.rm.setVelocity(-0.6)  # wiggle
            return

        if not self.path or self.idx >= len(self.path):
            self.lm.setVelocity(1.0); self.rm.setVelocity(-1.0)  # explore
            return

        x,y,yaw = self.pose()
        wx, wy = grid_to_world(*self.path[self.idx])
        if math.hypot(wx-x, wy-y) < LOOKAHEAD and self.idx < len(self.path)-1:
            self.idx += 1; wx, wy = grid_to_world(*self.path[self.idx])

        desired = math.atan2(wy - y, wx - x)
        err = wrap(desired - yaw)
        forward = BASE_SPEED * max(0.0, math.cos(err))
        ang = 5.5 * err
        L = clamp(forward - 0.08*ang, -MAX_WHEEL, MAX_WHEEL)
        R = clamp(forward + 0.08*ang, -MAX_WHEEL, MAX_WHEEL)
        self.lm.setVelocity(L); self.rm.setVelocity(R)

    def run(self):
        while self.r.step(DT) != -1:
            self.step()

if __name__ == "__main__":
    SwarmOneFile().run()
