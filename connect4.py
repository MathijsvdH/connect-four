# Shared code for the web app (app.py): a pure-NumPy Connect Four engine (OpenSpiel has no Windows wheels),
# the Q-networks and look-ahead search from the notebook, and the Double-DQN fine-tuning on recorded games.
import copy, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS, COLS = 6, 7


def pick_device():
    # CUDA if it actually works: a torch build without kernels for the GPU still reports is_available() == True
    if os.environ.get("C4_DEVICE"): return torch.device(os.environ["C4_DEVICE"])        # e.g. C4_DEVICE=cpu
    if torch.cuda.is_available():
        try:
            nn.Conv2d(1, 1, 3).cuda()(torch.zeros(1, 1, 6, 7, device="cuda")).sum().item(); return torch.device("cuda")
        except Exception: pass
    return torch.device("cpu")

DEVICE = pick_device()

def warm_up():
    # the first backward pass on a GPU loads its kernels (~25 s on an RTX 5060), later ones take milliseconds;
    # app.py runs this in a background thread at start-up so the first game's learning step is not slow
    if DEVICE.type != "cuda": return
    net = DuelResQNet().to(DEVICE)
    F.smooth_l1_loss(net(torch.zeros(64, 3, 6, 7, device=DEVICE)), torch.zeros(64, 7, device=DEVICE)).backward()
    torch.cuda.synchronize()


# ---- Game engine
# Implements the subset of the pyspiel state API the notebook uses (current_player, legal_actions, child,
# is_terminal, returns), so the search below is the same code as in the notebook (section 9.3).
# grid: 0 = empty, 1 = player 0 (moves first), 2 = player 1; row 0 = bottom row.
class Board:
    def __init__(self, moves=()):
        self.grid = np.zeros((ROWS, COLS), np.int8)
        self.height = [0] * COLS
        self.player, self.winner, self.moves, self.win_cells = 0, None, [], []
        for a in moves: self.apply_action(a)

    def clone(self):
        b = Board.__new__(Board)
        b.grid, b.height, b.player, b.winner = self.grid.copy(), self.height[:], self.player, self.winner
        b.moves, b.win_cells = self.moves[:], self.win_cells[:]
        return b

    def current_player(self): return self.player
    def is_terminal(self): return self.winner is not None or len(self.moves) == ROWS * COLS
    def legal_actions(self): return [] if self.is_terminal() else [c for c in range(COLS) if self.height[c] < ROWS]
    def returns(self):
        if self.winner is None: return [0.0, 0.0]
        return [1.0, -1.0] if self.winner == 0 else [-1.0, 1.0]

    def apply_action(self, c):
        r = self.height[c]
        self.grid[r, c] = self.player + 1; self.height[c] += 1; self.moves.append(int(c))
        line = self._four(r, c)
        if line: self.winner, self.win_cells = self.player, line
        self.player = 1 - self.player

    def child(self, c):
        b = self.clone(); b.apply_action(c)
        return b

    def _four(self, r, c):
        # cells of a 4-in-a-row through the disc just placed at (r, c), or [] if there is none
        me = self.grid[r, c]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            line = [(r, c)]
            for sgn in (1, -1):
                rr, cc = r + sgn * dr, c + sgn * dc
                while 0 <= rr < ROWS and 0 <= cc < COLS and self.grid[rr, cc] == me:
                    line.append((rr, cc)); rr += sgn * dr; cc += sgn * dc
            if len(line) >= 4: return line
        return []


# ---- Observation encoding (identical to encode / legal_mask in the notebook)
# row0_bottom comes from the exported checkpoint, so the board is never fed to the network upside down.
def encode(b, row0_bottom=True):
    # (3,6,7) float32: [my discs, opponent discs, empty], from the view of the player to move
    me = b.player + 1
    g = b.grid if row0_bottom else b.grid[::-1]
    return np.stack([g == me, g == 3 - me, g == 0]).astype(np.float32)

def legal_mask(b):
    m = np.zeros(COLS, dtype=bool); m[b.legal_actions()] = True
    return m


# ---- Q-networks (copies of the notebook classes; parameter names must match to load the weights)
class QNet(nn.Module):
    def __init__(self, ch=64, hidden=128):
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.fc1 = nn.Linear(ch * 6 * 7, hidden)
        self.out = nn.Linear(hidden, 7)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x)); x = F.relu(self.c3(x))
        return self.out(F.relu(self.fc1(x.flatten(1))))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))

class DuelResQNet(nn.Module):
    def __init__(self, ch=64, blocks=4, hidden=256):
        super().__init__()
        self.stem = nn.Conv2d(3, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.fc = nn.Linear(ch * 6 * 7, hidden)
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, 7)

    def forward(self, x):
        h = F.relu(self.fc(self.blocks(F.relu(self.stem(x))).flatten(1)))
        a = self.adv(h)
        return self.value(h) + a - a.mean(1, keepdim=True)

NETS = {"small": QNet, "resnet": DuelResQNet}


# ---- The AI player: a trained network + look-ahead search
# Checkpoints are written by export_agent in the notebook (section 10): weights + net type + gamma + board orientation.
class AI:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.ck, self.gamma, self.row0_bottom = ck, ck["gamma"], ck["row0_bottom"]
        self.q = NETS[ck["net"]]()
        self.q.load_state_dict(ck["weights"]); self.q.to(DEVICE).eval()

    def save(self, path, **extra):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ck = dict(self.ck, weights={k: v.detach().cpu().clone() for k, v in self.q.state_dict().items()}, **extra)
        torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)   # atomic: never leaves a half-written model

    @torch.no_grad()
    def q_values(self, b):
        return self.q(torch.as_tensor(encode(b, self.row0_bottom), device=DEVICE).unsqueeze(0))[0].cpu().numpy()

    @torch.no_grad()
    def state_values(self, states):
        # V(s) = max over legal Q(s, a) (clipped to [-1, 1]), from the view of the player to move in s (one batched forward pass)
        x = torch.as_tensor(np.stack([encode(s, self.row0_bottom) for s in states]), device=DEVICE)
        q = self.q(x).cpu().numpy()
        q[~np.stack([legal_mask(s) for s in states])] = -np.inf
        return np.clip(q.max(1), -1, 1)

    def move_scores(self, s, depth):
        # negamax look-ahead: score of every column for the player to move (-inf = illegal)
        p, sc, open_ = s.current_player(), np.full(7, -np.inf), []
        for a in s.legal_actions():
            c = s.child(a)
            if c.is_terminal():
                sc[a] = c.returns()[p]
                if sc[a] > 0: return sc                                # immediate win: no need to search further
            else: open_.append((a, c))
        if not open_: return sc
        if depth <= 1: vals = -self.gamma * self.state_values([c for _, c in open_])
        else:          vals = [-self.gamma * self.move_scores(c, depth - 1).max() for _, c in open_]
        for (a, _), v in zip(open_, vals): sc[a] = v
        return sc

    def choose(self, b, depth):
        # depth 0 = greedy network move; ties in the search score are broken by the raw Q-value
        q = self.q_values(b)
        sc = np.full(7, -np.inf); sc[b.legal_actions()] = q[b.legal_actions()]
        if depth > 0: sc = self.move_scores(b, depth)
        best = np.flatnonzero(sc >= sc.max() - 1e-9)
        return int(best[np.argmax(q[best])]), sc


# ---- Game log: one JSON object per line (append-only, easy to inspect or reload in the notebook)
def save_game(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f: f.write(json.dumps(record) + "\n")

def load_games(path):
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]


# ---- Learning from played games
# Every move of a recorded game (the human's too) becomes a transition from the mover's view, exactly like a
# self-play move in the notebook's train(), plus its left-right mirror. Off-policy Q-learning can learn from the
# human's moves because the target uses the best next move, not the move that was actually played.
def transitions(moves, row0_bottom):
    b, out = Board(), []
    for a in moves:
        me, obs = b.current_player(), encode(b, row0_bottom)
        b.apply_action(a)
        done = b.is_terminal()
        r = float(b.returns()[me]) if done else 0.0
        if done: nobs, nmask = np.zeros((3, 6, 7), np.float32), np.zeros(7, bool)
        else:    nobs, nmask = encode(b, row0_bottom), legal_mask(b)
        out.append((obs, a, r, nobs, nmask, float(done)))
        out.append((obs[:, :, ::-1], 6 - a, r, nobs[:, :, ::-1], nmask[::-1], float(done)))   # mirrored copy
    return out

def finetune(ai, games, steps=100, batch=64, lr=1e-4, target_every=25, seed=None):
    # Double-DQN updates (same loss as DQNAgent.learn) on all recorded games; half of every batch comes from
    # the latest game so a fresh loss is learned from at once, the other half keeps older games in memory.
    data = [transitions(g, ai.row0_bottom) for g in games if len(g) > 0]
    if not data: return None
    rng = np.random.RandomState(seed)
    T = [np.stack([np.array(t[k]) for d in data for t in d]) for k in range(6)]   # np.array also copies the mirrored views
    obs, act, rew, nobs, nmask, done = [torch.as_tensor(x, device=DEVICE) for x in T]   # all games fit on the GPU at once
    n_all, n_new = len(act), len(data[-1])
    target = copy.deepcopy(ai.q)
    opt = torch.optim.Adam(ai.q.parameters(), lr=lr)
    ai.q.train(); losses = []
    for step in range(steps):
        idx = torch.as_tensor(np.concatenate([n_all - n_new + rng.randint(0, n_new, batch // 2), rng.randint(0, n_all, batch - batch // 2)]), device=DEVICE)
        q = (ai.q(obs[idx]) * F.one_hot(act[idx], 7)).sum(1)
        with torch.no_grad():
            a_star = ai.q(nobs[idx]).masked_fill(~nmask[idx], -1e9).argmax(1, keepdim=True)
            y = rew[idx] - ai.gamma * target(nobs[idx]).gather(1, a_star).squeeze(1) * (1.0 - done[idx])
        loss = F.smooth_l1_loss(q, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(ai.q.parameters(), 10.0)
        opt.step(); losses.append(float(loss.item()))
        if (step + 1) % target_every == 0: target.load_state_dict(ai.q.state_dict())   # lets the result travel back several moves
    ai.q.eval()
    return float(np.mean(losses))
