import logging
import os
import sys
import random
import heapq
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from rl import pipe_io
from rl.logging_utils import setup_logging, DEFAULT_LOG_FILE

# Configure logging once for the whole module.
setup_logging()
logger = logging.getLogger(__name__)

# Visualization for "play" mode
# Hide pygame support prompt to avoid polluting stdout in pipe mode.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except Exception:
    pygame = None

# ============================
# CONFIG
# ============================

GRID_SIZE = 32
CELL_SIZE = 20

LOCAL_RADIUS = 2             # visibility 5×5 around catcher
VISIBILITY_RADIUS = 2        # visibility zone (Chebyshev) for Runner and Chatchers

WALL_SPAWN_CHANCE = 0.10    # chance of wall on the empty sells
MAX_TURNS = 200              # draw after MAX_TURNS steps of runner
NUM_EXITS = 3
NUM_CATCHERS = 3

EPISODES = 3000             # amount of episodes for training DQN-agent
SEED = 42
ID_BITS = 3                  # fixed size of catcher's one-hot ID

# Minimum BFS-distance from runner to closest exit when spawning
# For training steps:
# A: 8, B: 6/7, C1: 6, C2: 4, C3: 0
MIN_SPAWN_BFS = 6

# Folders
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Colors for game simulation
COLOR_WALL    = (100, 100, 100)
COLOR_EXIT    = (0, 255, 0)
COLOR_RUNNER  = (0, 0, 255)
COLOR_CATCHER = (255, 0, 0)
COLOR_BLOCK   = (150, 75, 0)
COLOR_BG      = (240, 240, 240)
COLOR_VIS     = (200, 200, 255, 50)

# ============================
# DQN: Double + Dueling
# ============================

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden = 512
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        self.val = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        f = self.feature(x)
        A = self.adv(f)
        V = self.val(f)
        return V + A - A.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.memory = deque(maxlen=100_000)
        self.gamma = 0.99
        self.n = 3
        self.n_buffer = deque(maxlen=self.n)

        # Hyperparameters
        self.lr = 1e-4
        self.batch_size = 128
        self.learn_starts = 2000
        self.learn_every = 4
        self.target_update_every = 2000

        self.step_count = 0
        self.epsilon = 1.0
        self.eps_end = 0.05
        self.eps_decay_steps = 2_000_000

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target(hard=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

    def update_target(self, hard=False, tau=0.01):
        if hard:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            with torch.no_grad():
                for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def act(self, state, valid_mask=None, eval_mode=False):
        """
        eval_mode = False  -> ε-greedy strategy (training)
        eval_mode = True   -> greedy strategy (exploration only)
        """
        # Greedy algorithm
        if eval_mode:
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q = self.model(s)
                if valid_mask is not None:
                    if isinstance(valid_mask, np.ndarray):
                        mask = torch.from_numpy(valid_mask.astype(np.bool_)).to(self.device)
                    else:
                        mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
                    if mask.sum().item() == 0:
                        return self.action_size - 1  # STAY
                    q = q.masked_fill((~mask).unsqueeze(0), -1e9)
            return int(q.argmax(dim=1).item())

        # Training: ε-greedy strategy
        self.step_count += 1
        self.epsilon = max(self.eps_end, 1.0 - self.step_count / self.eps_decay_steps)

        if random.random() < self.epsilon:
            if valid_mask is not None:
                valid_idx = np.flatnonzero(valid_mask)
                return int(random.choice(valid_idx)) if len(valid_idx) > 0 else random.randrange(self.action_size)
            return random.randrange(self.action_size)

        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(s)
            if valid_mask is not None:
                if isinstance(valid_mask, np.ndarray):
                    mask = torch.from_numpy(valid_mask.astype(np.bool_)).to(self.device)
                else:
                    mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
                if mask.sum().item() == 0:
                    return self.action_size - 1
                q = q.masked_fill((~mask).unsqueeze(0), -1e9)
        return int(q.argmax(dim=1).item())

    # n-step methods
    def _append_nstep(self, transition):
        self.n_buffer.append(transition)
        if len(self.n_buffer) < self.n:
            return None
        R = 0.0
        for i, (_, _, r, _, _) in enumerate(self.n_buffer):
            R += (self.gamma ** i) * r
        s0, a0, _, _, _ = self.n_buffer[0]
        _, _, _, ns, d = self.n_buffer[-1]
        return (s0, a0, R, ns, d)

    def remember(self, s, a, r, ns, done):
        t = (s, a, r, ns, done)
        nstep_t = self._append_nstep(t)
        if nstep_t is not None:
            self.memory.append(nstep_t)

    def replay(self):
        if len(self.memory) < max(self.learn_starts, self.batch_size):
            return
        if self.step_count % self.learn_every != 0:
            return

        batch = random.sample(self.memory, self.batch_size)
        s = torch.from_numpy(np.array([e[0] for e in batch], dtype=np.float32)).to(self.device)
        a = torch.tensor([e[1] for e in batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(self.device)
        ns = torch.from_numpy(np.array([e[3] for e in batch], dtype=np.float32)).to(self.device)
        d = torch.tensor([e[4] for e in batch], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_best = self.model(ns).argmax(dim=1, keepdim=True)
            next_q = self.target_model(ns).gather(1, next_best).squeeze(1)
            target = r + (1 - d) * (self.gamma ** self.n) * next_q

        q = self.model(s).gather(1, a).squeeze(1)
        loss = self.loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        if self.step_count % self.target_update_every == 0:
            self.update_target(hard=False)

    def flush_nstep(self):
        while len(self.n_buffer) > 0:
            R = 0.0
            for i, (_, _, r, _, _) in enumerate(self.n_buffer):
                R += (self.gamma ** i) * r
            s0, a0, _, _, _ = self.n_buffer[0]
            _, _, _, ns_last, d_last = self.n_buffer[-1]
            self.memory.append((s0, a0, R, ns_last, d_last))
            self.n_buffer.popleft()


# ============================
# GAME LOGIC
# ============================

# 9 actions: 4 steps, 4 blockages, stay action
ACTIONS = [
    (0, -1, 'move'),  # UP
    (1,  0, 'move'),  # RIGHT
    (0,  1, 'move'),  # DOWN
    (-1, 0, 'move'),  # LEFT
    (0, -1, 'block'), # BLOCK UP
    (1,  0, 'block'), # BLOCK RIGHT
    (0,  1, 'block'), # BLOCK DOWN
    (-1, 0, 'block'), # BLOCK LEFT
    (0,  0, 'stay')   # STAY
]

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunnerAgentBase:
    def get_action(self, game) -> tuple[int, int]:
        raise NotImplementedError


class HumanRunner(RunnerAgentBase):
    def get_action(self, game) -> tuple[int, int]:
        if pygame is None:
            return (0, 0)
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:    action = (0, -1)
                    elif event.key == pygame.K_DOWN:  action = (0, 1)
                    elif event.key == pygame.K_LEFT:  action = (-1, 0)
                    elif event.key == pygame.K_RIGHT: action = (1, 0)
        return action


class AStarRunner(RunnerAgentBase):
    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def get_action(self, game) -> tuple[int, int]:
        runner_pos = game.runner_pos
        visible = set(game.get_visibility(runner_pos, VISIBILITY_RADIUS))

        exits_vis = [e for e in game.exits if e in visible]
        if not exits_vis:
            moves = game._get_valid_runner_moves(runner_pos)
            if moves:
                return random.choice(moves)
            return (0, 0)

        goal = min(exits_vis, key=lambda e: self.heuristic(runner_pos, e))

        open_set = []
        heapq.heappush(open_set, (self.heuristic(runner_pos, goal), 0, runner_pos))
        came_from = {}
        g = {runner_pos: 0}

        while open_set:
            _, cost, cur = heapq.heappop(open_set)
            if cur == goal:
                while came_from.get(cur) not in (None, runner_pos) and cur != runner_pos:
                    cur = came_from[cur]
                dx = cur[0] - runner_pos[0]
                dy = cur[1] - runner_pos[1]
                if abs(dx)+abs(dy)==1:
                    return (dx, dy)
                break

            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                nx, ny = cur[0]+dx, cur[1]+dy
                nxt = (nx, ny)
                if nxt in visible and game._cell_free_for_runner(nxt):
                    nc = cost + 1
                    if nxt not in g or nc < g[nxt]:
                        g[nxt] = nc
                        came_from[nxt] = cur
                        pr = nc + self.heuristic(nxt, goal)
                        heapq.heappush(open_set, (pr, nc, nxt))

        moves = game._get_valid_runner_moves(runner_pos)
        if moves:
            return random.choice(moves)
        return (0, 0)


class EscapeGame:
    """
    Roles:
      - Runner: human or A*; without training.
      - Catchers: DQN (with training).
    Order of steps: runner, catcher1, catcher2, catcher3, runner, ...
    """
    def __init__(self, runner_type="astar", train_mode=False, seed=None, load_dqn_for_play=False):
        self.rng = random.Random(seed if seed is not None else random.randint(0, 10**9))
        self.train_mode = train_mode

        # field
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)  # 0-empty, 1-wall, 2-block
        self.exits = []
        self.catchers = []
        self.runner_pos = None

        # shared map: -1 unknown, 0 empty, 1 wall, 2 block, 3 exit
        self.shared_map = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)
        self.last_seen_runner = None  # (x, y) or None

        # progress/counters
        self.turn = 0  # 0 - runner; 1..NUM_CATCHERS - catchers' indexes+1
        self.runner_turns = 0
        self.game_over = False
        self.winner_text = None   # "Catchers wins" / "Runner wins" / "Draw"
        self.result = None        # 'catchers' / 'runner' / 'draw'

        # agents
        self.runner_agent = HumanRunner() if runner_type == "human" else AStarRunner()

        self.dqn_catchers = None
        self.state_size = self._dqn_state_size()
        self.action_size = len(ACTIONS)

        if train_mode or load_dqn_for_play:
            self.dqn_catchers = DQNAgent(self.state_size, self.action_size)
            if load_dqn_for_play:
                self.dqn_catchers.epsilon = 0.0
                best_path = os.path.join(MODELS_DIR, "dqn_best.pth")
                if os.path.exists(best_path):
                    try:
                        chk = torch.load(best_path, map_location='cpu')
                        self.dqn_catchers.model.load_state_dict(chk['model_state_dict'])
                        self.dqn_catchers.update_target(hard=True)
                        logger.info("Loaded dqn_best.pth")
                    except RuntimeError as e:
                        logger.warning(
                            "Failed to load dqn_best.pth (most likely, a different architecture). "
                            "PyTorch message: %s. Continue with the untrained model.",
                            e,
                        )
                else:
                    logger.warning("models/dqn_best.pth not found — playing without a trained model")

        self.blocked_exits = set()

        self._setup_board()
        self._refresh_shared_map_all()

    # map generation
    def _setup_board(self):
        # wall borders
        self.grid[0, :] = 1; self.grid[-1, :] = 1
        self.grid[:, 0] = 1; self.grid[:, -1] = 1

        # walls inside with chance = WALL_SPAWN_CHANCE
        for y in range(1, GRID_SIZE-1):
            for x in range(1, GRID_SIZE-1):
                if self.rng.random() < WALL_SPAWN_CHANCE:
                    self.grid[y, x] = 1

        # exits
        self.exits = []
        attempts = 0
        while len(self.exits) < NUM_EXITS and attempts < 5000:
            x = self.rng.randint(1, GRID_SIZE-2)
            y = self.rng.randint(1, GRID_SIZE-2)
            if self.grid[y, x] == 0 and (x, y) not in self.exits:
                self.exits.append((x, y))
            attempts += 1

        # catchers
        self.catchers = self._place_random(NUM_CATCHERS, avoid=set(self.exits))

        # runner with minimum BFS control before exit
        avoid = set(self.exits) | set(self.catchers)
        self.runner_pos = self._place_random(1, avoid=avoid)[0]
        min_bfs = self._runner_to_nearest_exit_dist()
        tries = 0
        while (min_bfs is None or (MIN_SPAWN_BFS and (min_bfs < MIN_SPAWN_BFS))) and tries < 5000:
            avoid = set(self.exits) | set(self.catchers)
            self.runner_pos = self._place_random(1, avoid=avoid)[0]
            min_bfs = self._runner_to_nearest_exit_dist()
            tries += 1

    def _place_random(self, count, avoid=None):
        avoid = set() if avoid is None else set(avoid)
        placed = []
        tries = 0
        while len(placed) < count and tries < 5000:
            x = self.rng.randint(1, GRID_SIZE-2)
            y = self.rng.randint(1, GRID_SIZE-2)
            if self.grid[y, x] == 0 and (x, y) not in avoid and (x, y) not in placed:
                placed.append((x, y))
                avoid.add((x, y))
            tries += 1
        return placed

    # utilities
    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def get_visibility(self, pos, radius=VISIBILITY_RADIUS):
        x, y = pos
        cells = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx, ny = x+dx, y+dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    cells.append((nx, ny))
        return cells

    def get_combined_catcher_visibility(self):
        vis = set()
        for c in self.catchers:
            vis.update(self.get_visibility(c))
        return vis

    def _cell_free_for_runner(self, pos):
        x, y = pos
        return (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and
                self.grid[y, x] == 0 and (x, y) not in self.catchers)

    def _valid_free(self, x, y):
        return (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and
                self.grid[y, x] == 0 and (x, y) != self.runner_pos and (x, y) not in self.catchers)

    def _get_valid_runner_moves(self, pos):
        x, y = pos
        moves = []
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if self._cell_free_for_runner((nx, ny)):
                moves.append((dx, dy))
        return moves

    def _runner_neighbors(self):
        x, y = self.runner_pos
        moves = []
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if self._cell_free_for_runner((nx, ny)):
                moves.append((nx, ny))
        return moves

    def is_runner_surrounded_by_blocks(self):
        x, y = self.runner_pos
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.grid[ny, nx] == 0:
                    return False
            else:
                continue
        return True

    def _bfs_distance(self, start, targets):
        if not targets:
            return None
        tx = set(targets)
        sx, sy = start
        if (sx, sy) in tx:
            return 0
        blocked = set(self.catchers)
        visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        q = deque([(sx, sy, 0)])
        visited[sy, sx] = True
        while q:
            x, y, d = q.popleft()
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                    continue
                if visited[ny, nx]:
                    continue
                if self.grid[ny, nx] != 0:
                    continue
                if (nx, ny) in blocked:
                    continue
                if (nx, ny) in tx:
                    return d+1
                visited[ny, nx] = True
                q.append((nx, ny, d+1))
        return None

    def _runner_to_nearest_exit_dist(self):
        d = self._bfs_distance(self.runner_pos, self.exits)
        return None if d is None else float(d)

    # shared map
    def _cell_base_type(self, x, y):
        """The basic type of cell without agents: 0 empty, 1 wall, 2 block, 3 exit."""
        if self.grid[y, x] == 1: return 1
        if self.grid[y, x] == 2: return 2
        if (x, y) in self.exits: return 3
        return 0

    def _update_shared_map_from_catcher(self, idx):
        """Update shared_map by 5x5 visibility of a specific catcher."""
        cx, cy = self.catchers[idx]
        for (x, y) in self.get_visibility((cx, cy), VISIBILITY_RADIUS):
            self.shared_map[y, x] = self._cell_base_type(x, y)
        vis = self.get_combined_catcher_visibility()
        if self.runner_pos in vis:
            self.last_seen_runner = self.runner_pos

    def _refresh_shared_map_all(self):
        """Recalculate shared_map by combining observations of all catchers."""
        for i in range(len(self.catchers)):
            self._update_shared_map_from_catcher(i)

    # DQN state
    def _dqn_state_size(self):
        channels = 7  # wall, block, exit, runner, other_catcher, self, unk
        W = 2*LOCAL_RADIUS + 1
        local = W*W*channels
        global_feats = 8 + ID_BITS
        return local + global_feats

    def _state_for_catcher(self, idx):
        cx, cy = self.catchers[idx]
        W = 2*LOCAL_RADIUS + 1
        C = 7  # wall, block, exit, runner, other_catcher, self, unk
        state = np.zeros((W, W, C), dtype=np.float32)

        combined_vis = set(self.get_combined_catcher_visibility())
        runner_is_visible = (self.runner_pos in combined_vis)

        for dy in range(-LOCAL_RADIUS, LOCAL_RADIUS+1):
            for dx in range(-LOCAL_RADIUS, LOCAL_RADIUS+1):
                x, y = cx+dx, cy+dy
                ix, iy = dx+LOCAL_RADIUS, dy+LOCAL_RADIUS

                if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
                    state[iy, ix, 0] = 1.0  # consider borders as a wall
                    continue

                cell = self.shared_map[y, x]  # -1,0,1,2,3
                if cell == -1:
                    state[iy, ix, 6] = 1.0  # unknown
                elif cell == 0:
                    pass
                elif cell == 1:
                    state[iy, ix, 0] = 1.0  # wall
                elif cell == 2:
                    state[iy, ix, 1] = 1.0  # block
                elif cell == 3:
                    state[iy, ix, 2] = 1.0  # exit

                if runner_is_visible and (x, y) == self.runner_pos:
                    state[iy, ix, 3] = 1.0

                if (x, y) in self.catchers and (x, y) != (cx, cy):
                    state[iy, ix, 4] = 1.0

        state[LOCAL_RADIUS, LOCAL_RADIUS, 5] = 1.0

        flat = state.flatten()

        rx, ry = self.runner_pos
        if runner_is_visible:
            rel_dx = (rx - cx) / GRID_SIZE
            rel_dy = (ry - cy) / GRID_SIZE
        else:
            rel_dx = 0.0
            rel_dy = 0.0
        runner_visible = 1.0 if runner_is_visible else 0.0

        if self.exits:
            ex = min(self.exits, key=lambda e: abs(e[0]-rx)+abs(e[1]-ry))
        else:
            ex = (rx, ry)
        ex_dx = (ex[0] - rx) / GRID_SIZE
        ex_dy = (ex[1] - ry) / GRID_SIZE

        runner_deg = len(self._runner_neighbors()) / 4.0
        dist_to_runner = (abs(rx - cx) + abs(ry - cy)) / (GRID_SIZE * 2.0)
        bf = self._runner_to_nearest_exit_dist()
        runner_exit_norm = (bf if bf is not None else (GRID_SIZE*2)) / (GRID_SIZE*2)

        id_bits = np.zeros(ID_BITS, dtype=np.float32)
        if idx < ID_BITS:
            id_bits[idx] = 1.0

        global_feats = np.array([rel_dx, rel_dy, runner_visible,
                                 ex_dx, ex_dy, runner_deg,
                                 dist_to_runner, runner_exit_norm], dtype=np.float32)
        return np.concatenate([flat, global_feats, id_bits], axis=0)

    # application of actions
    def _place_block(self, x, y):
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            if self.grid[y, x] == 0 and (x, y) != self.runner_pos and (x, y) not in self.catchers:
                self.grid[y, x] = 2
                return True
        return False

    def _apply_catcher_action(self, idx, action_idx):
        cx, cy = self.catchers[idx]
        dx, dy, kind = ACTIONS[action_idx]
        did_block = False
        invalid = False
        block_pos = None

        if kind == 'move':
            nx, ny = cx+dx, cy+dy
            if self._valid_free(nx, ny):
                self.catchers[idx] = (nx, ny)
            else:
                invalid = True
        elif kind == 'block':
            bx, by = cx+dx, cy+dy
            if self._place_block(bx, by):
                did_block = True
                block_pos = (bx, by)
            else:
                invalid = True
        else:
            pass  # stay
        return did_block, block_pos, invalid

    # reward
    def _reward_for_catcher(self, idx, did_block, block_pos, invalid, prev):
        reward = -0.8  # basic time penalty

        new_dist = self._manhattan(self.catchers[idx], self.runner_pos)
        if new_dist < prev['dist_to_runner']:
            reward += 7.0
        elif new_dist > prev['dist_to_runner']:
            reward -= 6.0

        if did_block:
            runner_vis = set(self.get_visibility(self.runner_pos))
            if block_pos in runner_vis:
                reward += 15.0
            else:
                reward -= 40.0

            if block_pos in self.exits and block_pos not in self.blocked_exits:
                reward += 150.0
                self.blocked_exits.add(block_pos)

            cur_reach = self._reachable_cells_for_runner()
            reduction = prev['reachable'] - cur_reach
            if reduction > 0:
                reward += min(120.0, 40.0 + 3.0 * reduction)
            else:
                reward -= 20.0

        if invalid:
            reward -= 15.0

        new_runner_exit = self._runner_to_nearest_exit_dist()
        p = prev['runner_exit_dist']
        if p is not None and new_runner_exit is not None:
            if new_runner_exit < p:
                reward -= 50.0
            elif new_runner_exit > p:
                inc = new_runner_exit - p
                reward += min(30.0, 10.0 + 5.0 * inc)

        cpos = self.catchers[idx]
        free_neighbors = sum(
            1 for dx,dy in [(0,1),(1,0),(0,-1),(-1,0)]
            if self._valid_free(cpos[0]+dx, cpos[1]+dy)
        )
        if free_neighbors <= 1:
            reward -= 50.0

        return reward

    def _reachable_cells_for_runner(self):
        x0, y0 = self.runner_pos
        visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        q = deque([(x0, y0)])
        visited[y0, x0] = True
        blocked = set(self.catchers)
        cnt = 0
        while q:
            x, y = q.popleft()
            cnt += 1
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE): continue
                if visited[ny, nx]: continue
                if self.grid[ny, nx] != 0: continue
                if (nx, ny) in blocked: continue
                visited[ny, nx] = True
                q.append((nx, ny))
        return cnt

    # action mask
    def _action_mask_for_catcher(self, idx):
        mask = np.zeros(len(ACTIONS), dtype=np.uint8)
        cx, cy = self.catchers[idx]

        def cell_free_for_catcher(x, y):
            return (
                0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and
                self.grid[y, x] == 0 and (x, y) != self.runner_pos and (x, y) not in self.catchers
            )

        for i, (dx, dy, kind) in enumerate(ACTIONS):
            if kind == 'move':
                nx, ny = cx + dx, cy + dy
                if cell_free_for_catcher(nx, ny):
                    mask[i] = 1
            elif kind == 'block':
                bx, by = cx + dx, cy + dy
                if cell_free_for_catcher(bx, by):
                    mask[i] = 1
            else:  # stay
                mask[i] = 1
        return mask

    # 1 step of the game
    def step(self):
        if self.game_over:
            return 0.0

        if self.runner_turns > MAX_TURNS:
            self.game_over = True
            self.winner_text = "Draw"
            self.result = 'draw'
            return 0.0

        # Runner action
        if self.turn == 0:
            self.runner_turns += 1
            dx, dy = self.runner_agent.get_action(self)
            moves = self._get_valid_runner_moves(self.runner_pos)
            if (dx, dy) in moves:
                nx, ny = self.runner_pos[0]+dx, self.runner_pos[1]+dy
                self.runner_pos = (nx, ny)
            elif moves:
                dx, dy = random.choice(moves)
                self.runner_pos = (self.runner_pos[0]+dx, self.runner_pos[1]+dy)

            if self.runner_pos in self.exits:
                self.game_over = True
                self.winner_text = "Runner wins"
                self.result = 'runner'
                return 0.0

            if self.is_runner_surrounded_by_blocks():
                self.game_over = True
                self.winner_text = "Catchers wins (runner is surrounded by blocks)"
                self.result = 'catchers'
                return 0.0

            self._refresh_shared_map_all()

            self.turn = 1
            return 0.0

        # Catchers action
        idx = self.turn - 1
        reward = 0.0

        self._refresh_shared_map_all()

        if self.dqn_catchers is not None:
            prev = {
                'dist_to_runner': self._manhattan(self.catchers[idx], self.runner_pos),
                'reachable': self._reachable_cells_for_runner(),
                'runner_exit_dist': self._runner_to_nearest_exit_dist()
            }
            state = self._state_for_catcher(idx)
            mask = self._action_mask_for_catcher(idx)
            action_idx = self.dqn_catchers.act(
                state,
                valid_mask=mask,
                eval_mode=not self.train_mode
            )

            did_block, block_pos, invalid = self._apply_catcher_action(idx, action_idx)

            self._update_shared_map_from_catcher(idx)

            terminal = False
            term_bonus = 0.0
            if self.is_runner_surrounded_by_blocks():
                terminal = True
                term_bonus = 1000.0
                self.game_over = True
                self.winner_text = "Catchers wins"
                self.result = 'catchers'

            if self.runner_pos in self.exits and not terminal:
                terminal = True
                term_bonus = -1000.0
                self.game_over = True
                self.winner_text = "Runner wins"
                self.result = 'runner'

            reward = self._reward_for_catcher(idx, did_block, block_pos, invalid, prev)
            reward += term_bonus

            next_state = self._state_for_catcher(idx)
            done = 1.0 if terminal else 0.0

            if self.train_mode:
                self.dqn_catchers.remember(state, action_idx, reward, next_state, done)
                self.dqn_catchers.replay()

            if terminal:
                self.dqn_catchers.update_target(hard=False)
                return reward

        else:
            rx, ry = self.runner_pos
            cx, cy = self.catchers[idx]
            best = None
            best_red = -1e9
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                nx, ny = cx+dx, cy+dy
                if self._valid_free(nx, ny):
                    old = self._manhattan((cx,cy), (rx,ry))
                    new = self._manhattan((nx,ny), (rx,ry))
                    red = old - new
                    if red > best_red:
                        best_red = red
                        best = (dx, dy)
            moved = False
            if best and self.rng.random() < 0.8:
                self.catchers[idx] = (cx+best[0], cy+best[1])
                moved = True
            if not moved:
                tx = 1 if rx > cx else -1 if rx < cx else 0
                ty = 1 if ry > cy else -1 if ry < cy else 0
                for dx, dy in [(tx,0),(0,ty),(tx,ty),(0,0)]:
                    if dx==0 and dy==0:
                        break
                    bx, by = cx+dx, cy+dy
                    if self._place_block(bx, by):
                        break

            if self.is_runner_surrounded_by_blocks():
                self.game_over = True
                self.winner_text = "Catchers wins"
                self.result = 'catchers'
                return 0.0

        self.turn = (self.turn + 1) % (NUM_CATCHERS + 1)
        return float(reward)

    # rendering
    def render(self, screen, clock=None, extra_status=""):
        if pygame is None:
            return
        screen.fill(COLOR_BG)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                r = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[y, x] == 1:
                    pygame.draw.rect(screen, COLOR_WALL, r)
                elif self.grid[y, x] == 2:
                    pygame.draw.rect(screen, COLOR_BLOCK, r)
                elif (x, y) in self.exits:
                    pygame.draw.rect(screen, COLOR_EXIT, r)
                pygame.draw.rect(screen, (220,220,220), r, 1)

        vis = self.get_combined_catcher_visibility()
        for vx, vy in vis:
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill(COLOR_VIS)
            screen.blit(s, (vx*CELL_SIZE, vy*CELL_SIZE))

        for c in self.catchers:
            pygame.draw.circle(
                screen, COLOR_CATCHER,
                (c[0]*CELL_SIZE + CELL_SIZE//2, c[1]*CELL_SIZE + CELL_SIZE//2),
                CELL_SIZE//3
            )
        pygame.draw.circle(
            screen, COLOR_RUNNER,
            (self.runner_pos[0]*CELL_SIZE + CELL_SIZE//2, self.runner_pos[1]*CELL_SIZE + CELL_SIZE//2),
            CELL_SIZE//3
        )

        font = pygame.font.SysFont(None, 24)
        status = f"Runner step: {self.runner_turns}/{MAX_TURNS} | {self.winner_text or 'Game in progress'}"
        text = font.render(status, True, (0,0,0))
        screen.blit(text, (10, GRID_SIZE*CELL_SIZE + 8))


# ============================
# TRAINING
# ============================

def train_dqn(episodes=EPISODES):
    set_seed(SEED)

    state_size = (2*LOCAL_RADIUS + 1)**2 * 7 + 8 + ID_BITS
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)

    resume_path = os.path.join(MODELS_DIR, "dqn_best.pth")
    if os.path.exists(resume_path):
        try:
            chk = torch.load(resume_path, map_location='cpu')
            agent.model.load_state_dict(chk['model_state_dict'])
            if 'optimizer_state_dict' in chk:
                agent.optimizer.load_state_dict(chk['optimizer_state_dict'])
            agent.step_count = chk.get('step_count', 0)
            agent.update_target(hard=True)
            logger.info("Resumed from %s (epsilon in ckpt: %s)", resume_path, chk.get('epsilon', 'n/a'))
        except Exception as e:
            logger.warning("Could not resume from %s: %s. Starting from scratch.", resume_path, e)

    best_total = -1e18
    results = {'catchers': 0, 'runner': 0, 'draw': 0}
    moving_avg = deque(maxlen=100)

    logger.info("TRAINING: %s episodes; device: %s, eps-> %s", episodes, agent.device, agent.eps_end)

    for ep in range(1, episodes + 1):
        game = EscapeGame(runner_type="astar", train_mode=True,
                          seed=random.randint(0, 10**9))
        game.dqn_catchers = agent

        total_reward, steps = 0.0, 0
        while not game.game_over:
            r = game.step()
            total_reward += r
            steps += 1
            if steps > 100000:
                game.game_over = True
                game.winner_text = "Draw"
                game.result = 'draw'

        agent.flush_nstep()

        moving_avg.append(total_reward)
        results[game.result] += 1

        if total_reward > best_total:
            best_total = total_reward
            path = os.path.join(MODELS_DIR, "dqn_best.pth")
            torch.save({
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'total_reward': total_reward,
                'step_count': agent.step_count,
            }, path)

        if ep % 50 == 0 or ep == 1:
            avg100 = sum(moving_avg)/len(moving_avg)
            logger.info(
                "[%5d/%d] R=%+8.1f avg100=%+8.1f steps=%4d -> %s",
                ep,
                episodes,
                total_reward,
                avg100,
                steps,
                game.winner_text,
            )

    logger.info("TRAINING COMPLETED!")
    logger.info(
        "Catchers wins: %d | Runner wins: %d | Draw: %d",
        results['catchers'],
        results['runner'],
        results['draw'],
    )
    logger.info("The best total reward: %+0.1f", best_total)
    logger.info("The best model is saved in models/dqn_best.pth")


# ============================
# EVALUATION
# ============================

def evaluate(num_episodes=300, runner_type="astar"):
    """
    Evaluates the current best model(models/dqn_best.pth) without training.
    The greedy-policy is used (eval_mode=True).
    """
    set_seed(SEED)

    state_size = (2*LOCAL_RADIUS + 1)**2 * 7 + 8 + ID_BITS
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)

    best_path = os.path.join(MODELS_DIR, "dqn_best.pth")
    if not os.path.exists(best_path):
        logger.warning("EVALUATION: The models/dqn_best.pth file was not found. Train the model first.")
        return

    try:
        chk = torch.load(best_path, map_location='cpu')
        agent.model.load_state_dict(chk['model_state_dict'])
        agent.update_target(hard=True)
        logger.info("EVALUATION: uploaded a model from %s (total_reward=%s)", best_path, chk.get('total_reward','?'))
    except Exception as e:
        logger.exception("Couldn't load the checkpoint for evaluate from %s", best_path)
        return

    agent.epsilon = 0.0
    agent.eps_end = 0.0

    stats = {'catchers': 0, 'runner': 0, 'draw': 0}
    rewards = []
    steps_list = []

    for ep in range(1, num_episodes + 1):
        game = EscapeGame(runner_type=runner_type,
                          train_mode=False,
                          seed=random.randint(0, 10**9),
                          load_dqn_for_play=False)
        game.dqn_catchers = agent

        total_reward, steps = 0.0, 0
        while not game.game_over:
            r = game.step()
            total_reward += r
            steps += 1
            if steps > 100000:
                game.game_over = True
                game.winner_text = "Draw (limit of steps in evaluate function)"
                game.result = 'draw'

        rewards.append(total_reward)
        steps_list.append(steps)
        stats[game.result] += 1

        if ep % 20 == 0 or ep == 1 or ep == num_episodes:
            wr_c = stats['catchers'] / ep
            wr_r = stats['runner'] / ep
            wr_d = stats['draw'] / ep
            avg_R = sum(rewards) / len(rewards)
            avg_steps = sum(steps_list) / len(steps_list)
            logger.info(
                "[EVAL %4d/%d] R_avg=%+7.1f | steps_avg=%5.1f | winC=%5.1f%% winR=%5.1f%% draw=%5.1f%%",
                ep,
                num_episodes,
                avg_R,
                avg_steps,
                wr_c*100,
                wr_r*100,
                wr_d*100,
            )

    logger.info("=== EVAL SUMMARY ===")
    logger.info("Episodes: %d", num_episodes)
    logger.info("Catchers wins: %d (%.1f%%)", stats['catchers'], stats['catchers']/num_episodes*100)
    logger.info("Runner wins:   %d (%.1f%%)", stats['runner'], stats['runner']/num_episodes*100)
    logger.info("Draw:          %d (%.1f%%)", stats['draw'], stats['draw']/num_episodes*100)
    logger.info("Average reward: %+0.2f", sum(rewards)/len(rewards))
    logger.info("Average length: %0.1f steps", sum(steps_list)/len(steps_list))

# ============================
# Pipe IO integration (Godot)
# ============================

PIPE_CELL_EMPTY = "_C"
PIPE_CELL_EXIT = "eC"
PIPE_CELL_OBSTACLE = "oC"
PIPE_CELL_RUNNER = "aeC"
PIPE_CELL_CATCHER = "acC"
PIPE_CELL_UNKNOWN = "unkC"

_PIPE_TOKEN_MAP = {
    "_c": PIPE_CELL_EMPTY,
    "empty": PIPE_CELL_EMPTY,
    "floor": PIPE_CELL_EMPTY,
    "ec": PIPE_CELL_EXIT,
    "exit": PIPE_CELL_EXIT,
    "oc": PIPE_CELL_OBSTACLE,
    "block": PIPE_CELL_OBSTACLE,
    "wall": PIPE_CELL_OBSTACLE,
    "ae": PIPE_CELL_RUNNER,
    "aec": PIPE_CELL_RUNNER,
    "runner": PIPE_CELL_RUNNER,
    "ac": PIPE_CELL_CATCHER,
    "acc": PIPE_CELL_CATCHER,
    "catcher": PIPE_CELL_CATCHER,
    "unk": PIPE_CELL_UNKNOWN,
    "unkc": PIPE_CELL_UNKNOWN,
    "unknown": PIPE_CELL_UNKNOWN,
}

_PIPE_DQN_AGENT = None


def _pipe_get_dqn_agent():
    global _PIPE_DQN_AGENT
    if _PIPE_DQN_AGENT is not None:
        return _PIPE_DQN_AGENT
    state_size = (2 * LOCAL_RADIUS + 1) ** 2 * 7 + 8 + ID_BITS
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)
    best_path = os.path.join(MODELS_DIR, "dqn_best.pth")
    if os.path.exists(best_path):
        try:
            chk = torch.load(best_path, map_location="cpu")
            agent.model.load_state_dict(chk["model_state_dict"])
            agent.update_target(hard=True)
            logger.info("[pipe] Loaded %s", best_path)
        except Exception as e:
            logger.warning("[pipe] Failed to load %s: %s", best_path, e)
    agent.epsilon = 0.0
    agent.eps_end = 0.0
    _PIPE_DQN_AGENT = agent
    return agent


def _pipe_normalize_token(tok):
    key = tok.strip()
    return _PIPE_TOKEN_MAP.get(key.lower(), key)


def _pipe_parse_rows(rows):
    parsed = []
    for raw in rows:
        toks = [_pipe_normalize_token(t) for t in raw.split() if t.strip()]
        if toks:
            parsed.append(toks)
    return parsed


def _pipe_build_game_from_rows(rows, meta=None):
    grid_tokens = _pipe_parse_rows(rows)
    if not grid_tokens:
        return None
    h = len(grid_tokens)
    w = len(grid_tokens[0])

    game = EscapeGame(
        runner_type="astar",
        train_mode=False,
        seed=0,
        load_dqn_for_play=False,
    )
    game.dqn_catchers = _pipe_get_dqn_agent()

    game.grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    game.exits = []
    game.runner_pos = (0, 0)

    for y in range(min(h, GRID_SIZE)):
        for x in range(min(w, GRID_SIZE)):
            token = grid_tokens[y][x]
            if token == PIPE_CELL_EXIT:
                game.exits.append((x, y))
                game.grid[y, x] = 0
            elif token == PIPE_CELL_OBSTACLE:
                game.grid[y, x] = 1
            elif token == PIPE_CELL_RUNNER:
                game.runner_pos = (x, y)
                game.grid[y, x] = 0
            elif token == PIPE_CELL_CATCHER:
                game.grid[y, x] = 0
                game.catchers.append((x, y))
            elif token == PIPE_CELL_UNKNOWN:
                game.grid[y, x] = 0  # always allow moving into unknown cells
            else:
                game.grid[y, x] = 0

    if meta and meta.get("role") and meta.get("role").startswith("C") and meta.get("pos"):
        cur_pos = (int(meta["pos_x"]), int(meta["pos_y"]))
        catcher_id = int(meta["role"][1:])
        try:
            index = game.catchers.index(cur_pos)
            game.catchers[index], game.catchers[catcher_id] = game.catchers[catcher_id], game.catchers[index]
        except ValueError:
            game.catchers[catcher_id] = (int(meta["pos_x"]), int(meta["pos_y"]))

    game.shared_map.fill(-1)
    for y in range(min(h, GRID_SIZE)):
        for x in range(min(w, GRID_SIZE)):
            token = grid_tokens[y][x]
            if token == PIPE_CELL_UNKNOWN:
                game.shared_map[y, x] = -1
            else:
                game.shared_map[y, x] = game._cell_base_type(x, y)

    game.last_seen_runner = None
    game._refresh_shared_map_all()

    return game


def _pipe_action_name(idx):
    dx, dy, kind = ACTIONS[idx]
    if kind == "move":
        if dx == 0 and dy == -1:
            return "move up"
        if dx == 1 and dy == 0:
            return "move right"
        if dx == 0 and dy == 1:
            return "move down"
        if dx == -1 and dy == 0:
            return "move left"
    if kind == "block":
        if dx == 0 and dy == -1:
            return "build up"
        if dx == 1 and dy == 0:
            return "build right"
        if dx == 0 and dy == 1:
            return "build down"
        if dx == -1 and dy == 0:
            return "build left"
    return "stay"


def _pipe_runner_decide(game):
    action_vec = game.runner_agent.get_action(game)
    dx, dy = action_vec
    # Validate against available moves; fall back to a safe move or stay.
    valid_moves = set(game._get_valid_runner_moves(game.runner_pos))
    if (dx, dy) not in valid_moves:
        if valid_moves:
            dx, dy = next(iter(valid_moves))
        else:
            dx, dy = (0, 0)

    if (dx, dy) == (0, -1):
        return "move up"
    if (dx, dy) == (1, 0):
        return "move right"
    if (dx, dy) == (0, 1):
        return "move down"
    if (dx, dy) == (-1, 0):
        return "move left"
    return "stay"


def _pipe_catcher_decide(game, idx):
    state = game._state_for_catcher(idx)
    mask = game._action_mask_for_catcher(idx)
    action_idx = game.dqn_catchers.act(state, valid_mask=mask, eval_mode=True)
    cx, cy = game.catchers[idx]

    def _is_valid_move(ai):
        dx, dy, kind = ACTIONS[ai]
        if kind == "move":
            return game._valid_free(cx + dx, cy + dy)
        if kind == "block":
            bx, by = cx + dx, cy + dy
            return (
                0 <= bx < GRID_SIZE and 0 <= by < GRID_SIZE and
                game.grid[by, bx] == 0 and (bx, by) != game.runner_pos and (bx, by) not in game.catchers
            )
        return True  # stay

    # If the chosen action is invalid, fall back to a valid one or stay.
    if not mask[action_idx] or not _is_valid_move(action_idx):
        valid_indices = [i for i, m in enumerate(mask) if m and _is_valid_move(i)]
        if valid_indices:
            action_idx = valid_indices[0]
        else:
            action_idx = len(ACTIONS) - 1  # stay

    act_name = _pipe_action_name(action_idx)
    logger.info(
        "[pipe] catcher=%d pos=(%d,%d) action=%s mask=%s",
        idx, cx, cy, act_name, mask.tolist()
    )
    return act_name


def _pipe_decider():
    def _parse_role(meta_role):
        raw = (meta_role or "").strip()
        if not raw:
            raw = "C1"
        lower = raw.lower()
        if lower.startswith("runner") or lower.startswith("r"):
            suffix = lower[1:] if lower.startswith("r") else lower[len("runner"):]
            idx_val = 0
            if suffix.isdigit():
                idx_val = max(int(suffix) - 1, 0)
            label = raw
            if not label:
                label = "R"
            return "runner", idx_val, label

        # catcher variants: "catcherX" or "cX" or fallback
        suffix = ""
        if lower.startswith("catcher"):
            suffix = lower[len("catcher"):]
        elif lower.startswith("c"):
            suffix = lower[1:]
        idx_val = max(int(suffix) - 1, 0) if suffix.isdigit() else 0
        label = raw if raw else f"C{idx_val+1}"
        return "catcher", idx_val, label

    def decide(rows, meta):
        role_type, idx_val, role_label = _parse_role(meta.get("role"))
        logger.info("[pipe] meta=%s role=%s idx=%d", meta or {}, role_type, idx_val)

        game = _pipe_build_game_from_rows(rows, meta=meta)
        if game is None:
            logger.warning("[pipe] Received empty/invalid grid; sending stay")
            return "stay"

        if role_type == "runner":
            act = _pipe_runner_decide(game)
            return f"{role_label} act {act}"
        if not game.catchers:
            return "stay"
        idx_val = max(0, min(idx_val, len(game.catchers) - 1))
        act = _pipe_catcher_decide(game, idx_val)
        return f"{role_label} act {act}"

    return decide


def run_pipe_agent(stream=None, out=None):
    """
    Pipe loop used by the Godot integration.
    - Godot sends a metadata line (role, id) and the vision grid rows.
    - We reconstruct an EscapeGame snapshot and use AStarRunner/DQNAgent for decisions.
    - Role (and catcher index) are read from the metadata role string, with optional id override.
    """
    log_path = os.path.abspath(DEFAULT_LOG_FILE)
    setup_logging(log_path)
    logger.info("[pipe] run_pipe_agent started; log=%s", log_path)
    pipe_io.run_loop(
        _pipe_decider(),
        stream=stream,
        out=out,
    )

# ============================
# GAME
# ============================

def play_loop():
    if pygame is None:
        logger.error("For visualization install pygame: pip install pygame")
        return

    mode = input("Runner: [1] A*  [2] Human: ").strip()
    runner_type = "human" if mode == "2" else "astar"

    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE + 36))
    pygame.display.set_caption("Escape RL — DQN Catchers vs Runner")
    clock = pygame.time.Clock()

    game = EscapeGame(runner_type=runner_type, train_mode=False,
                      seed=random.randint(0,10**9), load_dqn_for_play=True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game.game_over:
            game.step()

        eps = 0.0 if (game.dqn_catchers is None) else game.dqn_catchers.epsilon
        status = f"| eps={eps:.3f}"
        game.render(screen, clock=clock, extra_status=status)
        pygame.display.flip()
        clock.tick(8)

        if game.game_over:
            font = pygame.font.SysFont(None, 28)
            msg = font.render("SPACE — new game; ESC — exit", True, (0,0,0))
            screen.blit(msg, (10, GRID_SIZE*CELL_SIZE + 10))
            pygame.display.flip()
            wait = True
            while wait:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False; wait = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            game = EscapeGame(runner_type=runner_type, train_mode=False,
                                              seed=random.randint(0,10**9), load_dqn_for_play=True)
                            wait = False
                        elif event.key == pygame.K_ESCAPE:
                            running = False; wait = False

    pygame.quit()


if __name__ == "__main__":
    # Fallback for direct execution: default to play loop without parsing argv.
    play_loop()
