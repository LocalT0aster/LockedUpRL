# -*- coding: utf-8 -*-
import os
import sys
import random
import heapq
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# Визуализация используется только в режиме play
try:
    import pygame
except Exception:
    pygame = None

# ============================
# КОНФИГ
# ============================

GRID_SIZE = 32
CELL_SIZE = 20

# ОБЗОР И ВИДИМОСТЬ: 5x5
LOCAL_RADIUS = 2             # окно 5×5 вокруг ловца
VISIBILITY_RADIUS = 2        # зона видимости (Chebyshev) для раннера и ловцов

WALL_SPAWN_CHANCE = 0.08    # шанс стены на пустых клетках внутри
MAX_TURNS = 380              # ничья после MAX_TURNS ходов раннера
NUM_EXITS = 2
NUM_CATCHERS = 2

EPISODES = 9000             # количество эпизодов обучения
SEED = 42
ID_BITS = 3                  # фиксированный размер one-hot ID блокера (макс. число ловцов)

# Минимальная BFS-дистанция раннера до ближайшего выхода при спавне
# Меняй по этапам curriculum:
# A: 8, B: 6/7, C1: 6, C2: 4, C3: 0
MIN_SPAWN_BFS = 8

# Папки
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Цвета (pygame)
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
        self.n = 3                      # n-step
        self.n_buffer = deque(maxlen=self.n)

        # Гиперпараметры (меняй под этапы)
        self.lr = 1e-4
        self.batch_size = 128
        self.learn_starts = 2000
        self.learn_every = 4
        self.target_update_every = 2000

        self.step_count = 0
        self.epsilon = 1.0
        self.eps_end = 0.05
        self.eps_decay_steps = 2_000_000  # изменяй по плану A/B/C

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
        eval_mode = False  -> ε-жадная стратегия (обучение)
        eval_mode = True   -> чисто жадная стратегия (оценка / игра), без изменения epsilon/step_count
        """
        # ----- ЧИСТО ЖАДНЫЙ РЕЖИМ (оценка / игра) -----
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

        # ----- ОБУЧЕНИЕ: ε-жадная стратегия -----
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

    # ---- n-step вспомогательные методы ----
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
# Игровая логика
# ============================

# 9 действий: 4 шага, 4 блокировки, стоять
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
    def get_action(self, game):
        raise NotImplementedError


class HumanRunner(RunnerAgentBase):
    def get_action(self, game):
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

    def get_action(self, game):
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
    Роли:
      - Runner: человек или A*; не обучается.
      - Catchers: DQN (обучаются) или эвристика.
    Порядок ходов: runner, c1, c2, c3, runner, ...
    """
    def __init__(self, runner_type="astar", train_mode=False, seed=None, load_dqn_for_play=False):
        self.rng = random.Random(seed if seed is not None else random.randint(0, 10**9))
        self.train_mode = train_mode

        # поле
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)  # 0-пусто, 1-стена, 2-блок
        self.exits = []
        self.catchers = []
        self.runner_pos = None

        # shared map: -1 unknown, 0 empty, 1 wall, 2 block, 3 exit
        self.shared_map = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)
        self.last_seen_runner = None  # (x, y) или None

        # ход/счётчики
        self.turn = 0  # 0 - runner; 1..NUM_CATCHERS - индексы блокеров+1
        self.runner_turns = 0
        self.game_over = False
        self.winner_text = None   # "Победа Catchers" / "Победа Runner" / "Ничья"
        self.result = None        # 'catchers' / 'runner' / 'draw'

        # агенты
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
                        print("Загружена dqn_best.pth")
                    except RuntimeError as e:
                        print("[WARN] Не удалось загрузить dqn_best.pth (скорее всего, другая архитектура).")
                        print("       Сообщение PyTorch:", e)
                        print("       Продолжаем с не обученной моделью.")
                else:
                    print("Не найден models/dqn_best.pth — играем без обученной модели")

        self.blocked_exits = set()

        self._setup_board()
        self._refresh_shared_map_all()

    # ---------- генерация карты ----------
    def _setup_board(self):
        # границы стенами
        self.grid[0, :] = 1; self.grid[-1, :] = 1
        self.grid[:, 0] = 1; self.grid[:, -1] = 1

        # стены внутри с шансом
        for y in range(1, GRID_SIZE-1):
            for x in range(1, GRID_SIZE-1):
                if self.rng.random() < WALL_SPAWN_CHANCE:
                    self.grid[y, x] = 1

        # выходы
        self.exits = []
        attempts = 0
        while len(self.exits) < NUM_EXITS and attempts < 5000:
            x = self.rng.randint(1, GRID_SIZE-2)
            y = self.rng.randint(1, GRID_SIZE-2)
            if self.grid[y, x] == 0 and (x, y) not in self.exits:
                self.exits.append((x, y))
            attempts += 1

        # блокеры
        self.catchers = self._place_random(NUM_CATCHERS, avoid=set(self.exits))

        # раннер с контролем минимального BFS до выхода
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

    # ---------- утилиты ----------
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

    # ---------- shared map ----------
    def _cell_base_type(self, x, y):
        """Базовый тип клетки без учёта агентов: 0 пусто, 1 стена, 2 блок, 3 выход."""
        if self.grid[y, x] == 1: return 1
        if self.grid[y, x] == 2: return 2
        if (x, y) in self.exits: return 3
        return 0

    def _update_shared_map_from_catcher(self, idx):
        """Обновить shared_map по 5x5 видимости конкретного ловца."""
        cx, cy = self.catchers[idx]
        for (x, y) in self.get_visibility((cx, cy), VISIBILITY_RADIUS):
            self.shared_map[y, x] = self._cell_base_type(x, y)
        vis = self.get_combined_catcher_visibility()
        if self.runner_pos in vis:
            self.last_seen_runner = self.runner_pos

    def _refresh_shared_map_all(self):
        """Пересчитать shared_map объединением наблюдений всех ловцов."""
        for i in range(len(self.catchers)):
            self._update_shared_map_from_catcher(i)

    # ---------- DQN состояние ----------
    def _dqn_state_size(self):
        channels = 7  # wall, block, exit, runner, other_catcher, self, unk
        W = 2*LOCAL_RADIUS + 1      # 5
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
                    state[iy, ix, 0] = 1.0  # границы считаем стеной
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

    # ---------- применение действий ----------
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

    # ---------- награда ----------
    def _reward_for_catcher(self, idx, did_block, block_pos, invalid, prev):
        reward = -0.8  # базовый временной штраф

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

    # ---------- маска действий ----------
    def _action_mask_for_catcher(self, idx):
        mask = np.zeros(len(ACTIONS), dtype=np.uint8)
        cx, cy = self.catchers[idx]
        for i, (dx, dy, kind) in enumerate(ACTIONS):
            if kind == 'move':
                nx, ny = cx + dx, cy + dy
                if self._valid_free(nx, ny):
                    mask[i] = 1
            elif kind == 'block':
                bx, by = cx + dx, cy + dy
                if (0 <= bx < GRID_SIZE and 0 <= by < GRID_SIZE and
                    self.grid[by, bx] == 0 and (bx, by) != self.runner_pos and (bx, by) not in self.catchers):
                    mask[i] = 1
            else:  # stay
                mask[i] = 1
        return mask

    # ---------- один шаг игры ----------
    def step(self):
        if self.game_over:
            return 0.0

        if self.runner_turns > MAX_TURNS:
            self.game_over = True
            self.winner_text = "Ничья"
            self.result = 'draw'
            return 0.0

        # ====== ХОД RUNNER ======
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
                self.winner_text = "Победа Runner"
                self.result = 'runner'
                return 0.0

            if self.is_runner_surrounded_by_blocks():
                self.game_over = True
                self.winner_text = "Победа Catchers (runner окружён блоками)"
                self.result = 'catchers'
                return 0.0

            self._refresh_shared_map_all()

            self.turn = 1
            return 0.0

        # ====== ХОД CATCHERS ======
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
                self.winner_text = "Победа Catchers"
                self.result = 'catchers'

            if self.runner_pos in self.exits and not terminal:
                terminal = True
                term_bonus = -1000.0
                self.game_over = True
                self.winner_text = "Победа Runner"
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
            # простая эвристика
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
                self.winner_text = "Победа Catchers"
                self.result = 'catchers'
                return 0.0

        self.turn = (self.turn + 1) % (NUM_CATCHERS + 1)
        return float(reward)

    # ---------- отрисовка ----------
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
        status = f"Ход раннера: {self.runner_turns}/{MAX_TURNS} | {self.winner_text or 'Игра идёт'} {extra_status}"
        text = font.render(status, True, (0,0,0))
        screen.blit(text, (10, GRID_SIZE*CELL_SIZE + 8))


# ============================
# Тренировка
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
            print(f"Resumed from {resume_path} (epsilon in ckpt: {chk.get('epsilon','n/a')})")
        except Exception as e:
            print(f"[WARN] Could not resume from {resume_path}: {e}. Starting from scratch.")

    best_total = -1e18
    results = {'catchers': 0, 'runner': 0, 'draw': 0}
    moving_avg = deque(maxlen=100)

    print(f"ОБУЧЕНИЕ: {episodes} эпизодов; устройство: {agent.device}, eps-> {agent.eps_end}")

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
                game.winner_text = "Ничья"
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
            print(f"[{ep:5d}/{episodes}] R={total_reward:+8.1f} avg100={avg100:+8.1f} "
                  f"steps={steps:4d} -> {game.winner_text}")

    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Победа Catchers: {results['catchers']} | Победа Runner: {results['runner']} | Ничья: {results['draw']}")
    print(f"Лучшая суммарная награда: {best_total:+.1f}")
    print("Лучшая модель сохранена в models/dqn_best.pth")


# ============================
# Оценка (evaluate)
# ============================

def evaluate(num_episodes=300, runner_type="astar"):
    """
    Оценивает текущую лучшую модель (models/dqn_best.pth) без обучения.
    Используется greedy-политика (eval_mode=True).
    """
    set_seed(SEED)

    state_size = (2*LOCAL_RADIUS + 1)**2 * 7 + 8 + ID_BITS
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)

    best_path = os.path.join(MODELS_DIR, "dqn_best.pth")
    if not os.path.exists(best_path):
        print("ОЦЕНКА: файл models/dqn_best.pth не найден. Сначала потренируй модель.")
        return

    try:
        chk = torch.load(best_path, map_location='cpu')
        agent.model.load_state_dict(chk['model_state_dict'])
        agent.update_target(hard=True)
        print(f"ОЦЕНКА: загружена модель из {best_path} (total_reward={chk.get('total_reward','?')})")
    except Exception as e:
        print("[ОШИБКА] Не удалось загрузить чекпойнт для evaluate:", e)
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
                game.winner_text = "Ничья (лимит шагов в evaluate)"
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
            print(
                f"[EVAL {ep:4d}/{num_episodes}] "
                f"R_avg={avg_R:+7.1f} | steps_avg={avg_steps:5.1f} | "
                f"winC={wr_c*100:5.1f}% winR={wr_r*100:5.1f}% draw={wr_d*100:5.1f}%"
            )

    print("\n=== EVAL SUMMARY ===")
    print(f"Эпизодов: {num_episodes}")
    print(f"Победа Catchers: {stats['catchers']} ({stats['catchers']/num_episodes*100:.1f}%)")
    print(f"Победа Runner:   {stats['runner']} ({stats['runner']/num_episodes*100:.1f}%)")
    print(f"Ничья:           {stats['draw']} ({stats['draw']/num_episodes*100:.1f}%)")
    print(f"Средняя награда: {sum(rewards)/len(rewards):+.2f}")
    print(f"Средняя длина:   {sum(steps_list)/len(steps_list):.1f} шагов")


# ============================
# Pipe evaluation (Godot integration)
# ============================

CELL_EMPTY = "_C"
CELL_EXIT = "eC"
CELL_OBSTACLE = "oC"
CELL_RUNNER = "aeC"
CELL_CATCHER = "acC"
CELL_UNKNOWN = "unkC"

_PIPE_TOKEN_MAP = {
    "_c": CELL_EMPTY,
    "empty": CELL_EMPTY,
    "floor": CELL_EMPTY,
    "ec": CELL_EXIT,
    "exit": CELL_EXIT,
    "oc": CELL_OBSTACLE,
    "block": CELL_OBSTACLE,
    "wall": CELL_OBSTACLE,
    "ae": CELL_RUNNER,
    "aec": CELL_RUNNER,
    "runner": CELL_RUNNER,
    "ac": CELL_CATCHER,
    "acc": CELL_CATCHER,
    "catcher": CELL_CATCHER,
    "unk": CELL_UNKNOWN,
    "unkc": CELL_UNKNOWN,
    "unknown": CELL_UNKNOWN,
}

_MOVE_DIRS = [
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
]

_RUNNER_ACTIONS = {
    (0, -1): "move_up",
    (1, 0): "move_right",
    (0, 1): "move_down",
    (-1, 0): "move_left",
    (0, 0): "stay",
}

_CATCHER_MOVE_ACTIONS = {
    (0, -1): "move_up",
    (1, 0): "move_right",
    (0, 1): "move_down",
    (-1, 0): "move_left",
    (0, 0): "stay",
}

_CATCHER_BUILD_ACTIONS = {
    (0, -1): "build_up",
    (1, 0): "build_right",
    (0, 1): "build_down",
    (-1, 0): "build_left",
}


def _normalize_token(tok):
    key = tok.strip()
    lookup = _PIPE_TOKEN_MAP.get(key.lower())
    return lookup if lookup is not None else key


def _parse_grid_from_lines(lines):
    grid = []
    for raw in lines:
        row = [_normalize_token(tok) for tok in raw.split() if tok.strip()]
        if row:
            grid.append(row)
    return grid


def _find_positions(grid, targets):
    tgt = set(targets)
    found = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell in tgt:
                found.append((x, y))
    return found


def _in_bounds(grid, pos):
    x, y = pos
    return 0 <= y < len(grid) and 0 <= x < len(grid[y])


def _manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bfs_first_step(grid, start, goals, passable):
    if not goals:
        return None
    seen = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur in goals:
            while seen[cur] and seen[cur] != start:
                cur = seen[cur]
            dx = cur[0] - start[0]
            dy = cur[1] - start[1]
            return (dx, dy)
        for dx, dy in _MOVE_DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in seen:
                continue
            if not _in_bounds(grid, nxt):
                continue
            if not passable(nxt):
                continue
            seen[nxt] = cur
            q.append(nxt)
    return None


def _runner_pipe_action(grid, allow_unknown=False):
    runners = _find_positions(grid, {CELL_RUNNER})
    if not runners:
        return "stay"
    runner = runners[0]
    exits = _find_positions(grid, {CELL_EXIT})
    catchers = _find_positions(grid, {CELL_CATCHER})

    def passable(pos):
        cell = grid[pos[1]][pos[0]]
        if cell in (CELL_OBSTACLE, CELL_CATCHER):
            return False
        if cell == CELL_UNKNOWN and not allow_unknown:
            return False
        return True

    step = _bfs_first_step(grid, runner, set(exits), passable)
    if step is not None:
        return _RUNNER_ACTIONS.get(step, "stay")

    best_score = -1e18
    best_action = "stay"
    for dx, dy in _MOVE_DIRS + [(0, 0)]:
        nxt = (runner[0] + dx, runner[1] + dy)
        if not _in_bounds(grid, nxt):
            continue
        cell = grid[nxt[1]][nxt[0]]
        if cell in (CELL_OBSTACLE, CELL_CATCHER):
            continue
        if cell == CELL_UNKNOWN and not allow_unknown:
            continue
        dist_to_catcher = min((_manhattan_dist(nxt, c) for c in catchers), default=10)
        dist_to_exit = min((_manhattan_dist(nxt, e) for e in exits), default=0)
        score = dist_to_catcher * 2.0 - dist_to_exit
        if score > best_score:
            best_score = score
            best_action = _RUNNER_ACTIONS[(dx, dy)]
    return best_action


def _catcher_pipe_action(grid, catcher_index=0, allow_unknown=False):
    catchers = _find_positions(grid, {CELL_CATCHER})
    if not catchers:
        return "stay"
    idx = min(max(catcher_index, 0), len(catchers) - 1)
    catcher = catchers[idx]
    runner_positions = _find_positions(grid, {CELL_RUNNER})
    exits = _find_positions(grid, {CELL_EXIT})

    def passable(pos):
        cell = grid[pos[1]][pos[0]]
        if cell == CELL_OBSTACLE:
            return False
        if cell == CELL_UNKNOWN and not allow_unknown:
            return False
        if cell == CELL_CATCHER and pos != catcher:
            return False
        return True

    goals = runner_positions or exits
    step = _bfs_first_step(grid, catcher, set(goals), passable)
    if step is not None:
        return _CATCHER_MOVE_ACTIONS.get(step, "stay")

    if runner_positions:
        target = runner_positions[0]
        prioritized_dirs = sorted(
            _MOVE_DIRS,
            key=lambda d: _manhattan_dist((catcher[0] + d[0], catcher[1] + d[1]), target)
        )
        for dx, dy in prioritized_dirs:
            tx, ty = catcher[0] + dx, catcher[1] + dy
            if not _in_bounds(grid, (tx, ty)):
                continue
            cell = grid[ty][tx]
            if cell in (CELL_EMPTY, CELL_EXIT):
                return _CATCHER_BUILD_ACTIONS.get((dx, dy), "stay")

    if exits:
        step = _bfs_first_step(grid, catcher, set(exits), passable)
        if step is not None:
            return _CATCHER_MOVE_ACTIONS.get(step, "stay")

    for dx, dy in _MOVE_DIRS:
        nxt = (catcher[0] + dx, catcher[1] + dy)
        if _in_bounds(grid, nxt) and passable(nxt):
            return _CATCHER_MOVE_ACTIONS.get((dx, dy), "stay")
    return "stay"


def _read_pipe_observation(stream):
    """
    Reads a single observation from the stream.
    Format: optional metadata line (key=value ...), followed by grid rows.
    A blank line separates observations.
    """
    meta = {}
    rows = []
    while True:
        line = stream.readline()
        if line == "":
            break  # EOF
        stripped = line.strip()
        if stripped == "":
            if rows:
                break
            continue
        if not rows and "=" in stripped and all("=" in part for part in stripped.split()):
            for part in stripped.split():
                key, _, value = part.partition("=")
                if key:
                    meta[key.strip().lower()] = value.strip()
            continue
        rows.append(stripped)
    if not rows and not meta:
        return None, None
    return rows, meta


def run_pipe_agent(role="catcher", catcher_index=0, allow_unknown=False, stream=None, out=None):
    """
    Runs an interactive loop:
    - Reads a vision map from stdin (lines of tokens, blank line separates turns).
    - Outputs a single action per turn to stdout.
    Meta line example: "role=runner id=0".
    """
    stream = stream or sys.stdin
    out = out or sys.stdout
    while True:
        rows, meta = _read_pipe_observation(stream)
        if rows is None:
            break
        grid = _parse_grid_from_lines(rows)
        if not grid:
            print("stay", file=out, flush=True)
            continue
        meta = meta or {}
        chosen_role = meta.get("role", role).lower()
        idx_override = meta.get("id")
        idx_val = catcher_index
        if idx_override is not None:
            try:
                idx_val = int(idx_override)
            except ValueError:
                idx_val = catcher_index

        if chosen_role == "runner":
            action = _runner_pipe_action(grid, allow_unknown=allow_unknown)
        else:
            action = _catcher_pipe_action(grid, catcher_index=idx_val, allow_unknown=allow_unknown)
        print(action, file=out, flush=True)


# ============================
# Игра (play_loop)
# ============================

def play_loop():
    if pygame is None:
        print("Для визуализации установи pygame: pip install pygame")
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
            msg = font.render("SPACE — новая игра; ESC — выход", True, (0,0,0))
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


def main():
    raise RuntimeError("Command-line handling was moved to main.py. Use `python main.py <command>`.")

if __name__ == "__main__":
    # Fallback for direct execution: default to play loop without parsing argv.
    play_loop()
