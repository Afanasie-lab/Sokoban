"""
4-player Halma bot for Practical 3.
The engine calls AI_Player_Team29(board, player, flag) and expects
("A1", "B1") style references back according to the requirements.
flag=1 also writes the search tree to Team29_Tree.png.

Attribution
- The per-ply beam limits (6,5,4,3,3,2,2,2) follow Afanasie's work.
- The assignment-distance evaluation (minimum over permutations) is the
  same technique used in Afanasie's and Eloise's work.
- The reduced zone penalties (-5/-5) were inspired by Eloise's
  evaluation, which relies mostly on Manhattan distance to the goal,
  benchmarking her agent motivated keeping this search lightweight.
- The maxn search with immediate (multiplayer alpha-beta) pruning, the
  transposition table, the branch ordering, and the numba JIT core are
  Gecenio's work.
  
  1. Playing field as an integer in base 5
  Instead of moving matrices around just use bitmaps with an integer
  since it's 5x5 it has 25 cells and each is on range 0-4 so
  it is modular arithmetic in some way.
  TT is super fast because moving a piece is just sums and muls
  using precalculated powers of 5 so it's somehow like a Lookup Table.
  
  2. Numba
  I discovered this tool last year, we all know python is slow so numba is 
  a JIT tool which accelerates things.
  
  3. Frozenset
  Was doing leetcode and came across them, they allow O(1) searches
  so they are perfect for this use case and since they are immutable
  we ensure no errors or modifications to the winning cells.
  
"""

SIZE = 5
PLAYERS = (1, 2, 3, 4)

SEARCH_DEPTH = 2  # in rounds, every player moves twice = 8 plies
USE_PRUNING = True  # multiplayer alpha-beta pruning (bounds passed into maxn)
INF = 10 ** 9
MAX_TREE_NODES = 400
WIN = 10_000
CLAMP = 1000

# moves kept per ply, first ply to deepest basically it bounds the tree at approx 8k leaves ig
BRANCH_LIMITS = (6, 5, 4, 3, 3, 2, 2, 2)

# goal cells as flat indexes (0..24)
# win_cells_1v1
win_cells = {
    1: frozenset((19, 23, 24)),   # E4, D5, E5
    2: frozenset((15, 20, 21)),   # A4, A5, B5
    3: frozenset((0, 1, 5)),      # A1, B1, A2
    4: frozenset((3, 4, 9)),      # D1, E1, E2
}
win_cells_1v1 = {
    1: frozenset((19, 23, 24)),
    2: frozenset((0, 1, 5)),
    3: frozenset(),
    4: frozenset(),
}
active_zones = win_cells
goal_lists = {}


def set_zones(zones):
    global active_zones, goal_lists
    active_zones = zones
    goal_lists = {p: (tuple(sorted(z)) if z else (-1, -1, -1))
                  for p, z in zones.items()}


set_zones(win_cells)


def detect_zones(board):
    # the 1v1 test rig has no player 3/4 pieces
    ds = digits(board)
    if any(d in (3, 4) for d in ds):
        return win_cells
    return win_cells_1v1


def branch_limit(depth):
    if depth >= len(BRANCH_LIMITS):
        return BRANCH_LIMITS[0]
    return BRANCH_LIMITS[len(BRANCH_LIMITS) - depth]


# precomputed tables

pow5 = tuple(5 ** i for i in range(25))

ortho = ((-1, 0), (1, 0), (0, -1), (0, 1))
steps = tuple([] for _ in range(25))
jumps = tuple([] for _ in range(25))
for i in range(25):
    r, c = divmod(i, 5)
    for dr, dc in ortho:
        r1, c1 = r + dr, c + dc
        r2, c2 = r + 2 * dr, c + 2 * dc
        if 0 <= r1 < 5 and 0 <= c1 < 5:
            steps[i].append(r1 * 5 + c1)
            if 0 <= r2 < 5 and 0 <= c2 < 5:
                jumps[i].append((r1 * 5 + c1, r2 * 5 + c2))


manh_list = []
for i in range(25):
    row = []
    r1, c1 = i // 5, i % 5
    for j in range(25):
        r2, c2 = j // 5, j % 5
        row.append(abs(r1 - r2) + abs(c1 - c2))
    manh_list.append(tuple(row))

manhattan = tuple(manh_list)

perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


# board encoding

def to_ref(i):
    r, c = divmod(i, 5)
    return chr(ord('A') + c) + str(r + 1)


def from_ref(s):
    if not (isinstance(s, str) and len(s) == 2 and s[0] in 'ABCDE'
            and s[1] in '12345'):
        raise ValueError('bad cell reference: %r' % (s,))
    return (int(s[1]) - 1) * 5 + ord(s[0]) - ord('A')


def digits(board):
    return [(board // pow5[i]) % 5 for i in range(25)]


def apply_move(board, f, t, player):
    return board + player * (pow5[t] - pow5[f])


def to_grid(board):
    return [[(board // pow5[r * 5 + c]) % 5 for c in range(5)]
            for r in range(5)]


def validate(board, player):
    # encode the engine's input and check it, returns None after a warning = invalid
    if player not in PLAYERS:
        print('Warning: invalid player id %r (expected 1-4).' % (player,))
        return None
    try:
        rows = list(board)
    except Exception:
        rows = None
    if rows is None or len(rows) != 5:
        print('Warning: board does not have 5 rows.')
        return None
    counts = {p: 0 for p in PLAYERS}
    enc = 0
    for r in range(5):
        row = rows[r]
        if len(row) != 5:
            print('Warning: row %d is not 5 cells wide.' % r)
            return None
        for c in range(5):
            raw = row[c]
            try:
                v = int(raw)
            except (TypeError, ValueError):
                print('Warning: bad cell value %r.' % (raw,))
                return None
            if v != raw or not 0 <= v <= 4:
                print('Warning: cell %r is outside the range 0-4.' % (raw,))
                return None
            if v:
                counts[v] += 1
            enc += v * pow5[r * 5 + c]
    # pieces are never captured so 3 per present player, 0 for absent ones
    if any(counts[p] not in (0, 3) for p in PLAYERS) \
            or not any(counts[p] == 3 for p in PLAYERS):
        print('Warning: piece counts are wrong: %s.' % (counts,))
        return None
    return enc


def fallback_move(board, player):
    # invalid input then we just return some legal move instead of crashing the whole thing
    try:
        enc = 0
        for r in range(5):
            for c in range(5):
                enc += int(board[r][c]) * pow5[r * 5 + c]
        moves = legal_moves(enc, player)
        if moves:
            f, t = moves[0]
            return (to_ref(f), to_ref(t))
    except Exception:
        pass
    return (None, None)


# moves

def legal_moves(board, player):
    ds = digits(board)
    out = []
    for i in range(25):
        if ds[i] == player:
            for t in steps[i]:
                if ds[t] == 0:
                    out.append((i, t))
            for mid, land in jumps[i]:
                if ds[mid] != 0 and ds[land] == 0:
                    out.append((i, land))
    return out


def winner(ds):
    for p in PLAYERS:
        pieces = [i for i in range(25) if ds[i] == p]
        if len(pieces) == 3 and all(i in active_zones[p] for i in pieces):
            return p
    return None


def terminal_scores(w, depth):
    # depth scaled so faster wins score higher
    return tuple(WIN + depth if p == w else -WIN - depth for p in PLAYERS)


# evaluation

def assign_dist(pieces, goal):
    # cheapest assignment of the pieces to diff goal cells
    n = len(pieces)
    if n == 0:
        return 0
    if n == 3:
        best = 10 ** 9
        for p0, p1, p2 in perms:
            d = (manhattan[pieces[0]][goal[p0]]
                 + manhattan[pieces[1]][goal[p1]]
                 + manhattan[pieces[2]][goal[p2]])
            if d < best:
                best = d
        return best
    best = 10 ** 9
    if n == 1:
        for g in goal:
            best = min(best, manhattan[pieces[0]][g])
    else:
        for g0 in goal:
            for g1 in goal:
                if g1 != g0:
                    best = min(best, manhattan[pieces[0]][g0]
                               + manhattan[pieces[1]][g1])
    return best


def evaluate(ds):
    # mostly Manhattan distance to the own goal plus small zone terms
    pieces = {p: [] for p in PLAYERS}
    for i in range(25):
        d = ds[i]
        if d:
            pieces[d].append(i)

    own_zone = [0, 0, 0, 0]
    foreign_zone = [0, 0, 0, 0]
    blocked = [0, 0, 0, 0]
    for p in PLAYERS:
        for cell in pieces[p]:
            if cell in active_zones[p]:
                own_zone[p - 1] += 1
            elif any(cell in active_zones[q] for q in PLAYERS if q != p):
                foreign_zone[p - 1] += 1
        for q in PLAYERS:
            if q == p:
                continue
            for cell in pieces[q]:
                if cell in active_zones[p]:
                    blocked[p - 1] += 1

    scores = []
    for p in PLAYERS:
        dist = assign_dist(pieces[p], goal_lists[p])
        s = (-10 * dist + 20 * own_zone[p - 1] - 5 * foreign_zone[p - 1]
             - 5 * blocked[p - 1])
        scores.append(max(-CLAMP, min(CLAMP, s)))
    return tuple(scores)


# branch ordering

def order_moves(board, player, moves, ds, tt_move):
    # best first tt move, bigger distance gain, jumps, own goal
    pieces = [i for i in range(25) if ds[i] == player]
    goal = goal_lists[player]
    cur = assign_dist(pieces, goal)
    scored = []
    for f, t in moves:
        new_pieces = [t if x == f else x for x in pieces]
        gain = cur - assign_dist(new_pieces, goal)
        is_jump = manhattan[f][t] == 2
        own = t in active_zones[player]
        foreign = any(t in active_zones[q] for q in PLAYERS if q != player)
        scored.append((gain, is_jump, own, not foreign, f, t))
    scored.sort(reverse=True)
    ordered = [(s[4], s[5]) for s in scored]
    if tt_move is not None:
        ordered = ([m for m in ordered if m == tt_move]
                   + [m for m in ordered if m != tt_move])
    return ordered


# maxn

NODES = 0


def reset_nodes():
    global NODES
    NODES = 0


def node_count():
    return NODES


def maxn(board, player, depth, bounds, tt, viz, parent, label):
    """
    value vector search, one component per player. bounds = alpha-beta
    style pruning adapted for maxn: once our best beats the bound from
    the closest ancestor of the same player, the rest cannot matter
    (strict >, bounded scores). Nodes cut this way are not stored in tt.
    """

    global NODES
    NODES += 1
    

    ds = digits(board)
    w = winner(ds)
    if w is not None:
        v = terminal_scores(w, depth)
        record_node(viz, parent, label, v)
        return v, None
    if depth == 0:
        v = evaluate(ds)
        record_node(viz, parent, label, v)
        return v, None

    key = (board, player)
    entry = tt.get(key)
    if entry is not None and entry[0] >= depth:
        record_node(viz, parent, label + ' [tt]', entry[1])
        return entry[1], entry[2]

    moves = legal_moves(board, player)
    if not moves:
        v = evaluate(ds)
        record_node(viz, parent, label, v)
        return v, None

    nid = record_node(viz, parent, label, None)
    ordered = order_moves(board, player, moves, ds,
                          entry[2] if entry is not None else None)
    ordered = ordered[:branch_limit(depth)]

    best_v = None
    best_m = None
    complete = True
    bounds_out = bounds
    p = player - 1
    for m in ordered:
        nb = apply_move(board, m[0], m[1], player)
        v, _ = maxn(nb, player % 4 + 1, depth - 1, bounds_out, tt, viz,
                    nid, 'P%d: %s>%s' % (player, to_ref(m[0]), to_ref(m[1])))
        if best_v is None or v[p] > best_v[p]:
            best_v = v
            best_m = m
            bounds_out = (bounds_out[:p] + (max(bounds_out[p], v[p]),)
                          + bounds_out[p + 1:])
            if USE_PRUNING and v[p] > bounds[p]:
                complete = False
                break

    update_node(viz, nid, label, best_v)
    if complete:
        tt[key] = (depth, best_v, best_m)
    return best_v, best_m


# search tree output (flag=1)

class TreeRecorder:
    # collects the tree while maxn runs,parents before children
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes
        self.entries = {}
        self.insertion = []
        self.next_id = 1
        self.enabled = True

    def add(self, parent, tag):
        if not self.enabled:
            return None
        nid = self.next_id
        self.next_id += 1
        self.entries[nid] = [parent, tag if tag is not None else '']
        self.insertion.append(nid)
        if self.next_id > self.max_nodes:
            self.enabled = False
        return nid

    def update(self, nid, tag):
        if nid is not None and nid in self.entries:
            self.entries[nid][1] = tag

    def build(self):
        return self.entries, self.insertion


def record_node(viz, parent, label, value):
    if viz is None:
        return None
    tag = label if value is None else '%s | %s' % (label, value)
    return viz.add(parent, tag)


def update_node(viz, nid, label, value):
    if viz is not None and nid is not None:
        viz.update(nid, '%s | %s' % (label, value))


def render_tree(recorder, out_name='Team29_Tree.png'):
    # treelib -> dot file -> png fallsback to text files if that fails
    entries, order = recorder.build()
    try:
        from treelib import Tree
    except ImportError:
        print('Warning: treelib is not installed; skipping the tree image.')
        return
    tree = Tree()
    for nid in order:
        parent, tag = entries[nid]
        tree.create_node(tag, nid, parent=parent)
    print('Search tree: %d nodes recorded.' % len(order))
    dot_name = out_name.rsplit('.', 1)[0] + '.dot'
    txt_name = out_name.rsplit('.', 1)[0] + '.txt'
    try:
        tree.to_graphviz(filename=dot_name)
        import subprocess
        subprocess.run(['dot', '-Tpng', dot_name, '-o', out_name], check=True)
        print('Search tree saved to %s' % out_name)
    except Exception as exc:
        tree.save2file(txt_name)
        print('Warning: could not render the PNG (%s); saved %s and %s '
              'instead.' % (exc, dot_name, txt_name))

"""
jit version of the search (same ordering keys, pruning and limits, so it
picks the same moves) used when flag=0. numba needs a restricted subset
ints and numpy arrays only, no dicts/strings inside the compiled code.
"""
jit_core = None  # None = not loaded yet, False = numba missing


def init_jit():
    import numpy as np
    from numba import njit

    step_tbl = np.full((25, 4), -1, dtype=np.int64)
    jump_tbl = np.full((25, 4, 2), -1, dtype=np.int64)
    for i in range(25):
        for k, t in enumerate(steps[i]):
            step_tbl[i, k] = t
        for k, (mid, land) in enumerate(jumps[i]):
            jump_tbl[i, k, 0] = mid
            jump_tbl[i, k, 1] = land
    manh_tbl = np.array(manhattan, dtype=np.int64)
    perms_tbl = np.array(perms, dtype=np.int64)
    pow5_tbl = np.array(pow5, dtype=np.int64)
    limits_tbl = np.array(BRANCH_LIMITS, dtype=np.int64)

    OFF = 16384
    MASK = 32767
    BIG = 10 ** 9

    @njit(cache=True)
    def assign_core(ps, gs, n):
        best = 1000000000
        if n == 0:
            return 0
        if n == 3:
            for k in range(6):
                d = (manh_tbl[ps[0], gs[perms_tbl[k, 0]]]
                     + manh_tbl[ps[1], gs[perms_tbl[k, 1]]]
                     + manh_tbl[ps[2], gs[perms_tbl[k, 2]]])
                if d < best:
                    best = d
            return best
        if n == 1:
            for k in range(3):
                d = manh_tbl[ps[0], gs[k]]
                if d < best:
                    best = d
        else:
            for k0 in range(3):
                for k1 in range(3):
                    if k1 != k0:
                        d = manh_tbl[ps[0], gs[k0]] + manh_tbl[ps[1], gs[k1]]
                        if d < best:
                            best = d
        return best

    @njit(cache=True)
    def eval_core(pieces, counts, goal_tbl):
        # same score formula as evaluate(), packed into one int64
        pack = 0
        for p in range(4):
            n = counts[p]
            own = 0
            foreign = 0
            blocked = 0
            for k in range(n):
                cell = pieces[p, k]
                if (cell == goal_tbl[p, 0] or cell == goal_tbl[p, 1]
                        or cell == goal_tbl[p, 2]):
                    own += 1
                else:
                    for q in range(4):
                        if q != p and (cell == goal_tbl[q, 0]
                                       or cell == goal_tbl[q, 1]
                                       or cell == goal_tbl[q, 2]):
                            foreign += 1
            for q in range(4):
                if q != p:
                    for k in range(counts[q]):
                        cell = pieces[q, k]
                        if (cell == goal_tbl[p, 0] or cell == goal_tbl[p, 1]
                                or cell == goal_tbl[p, 2]):
                            blocked += 1
            if n == 0:
                s = 0
            else:
                d = assign_core(pieces[p], goal_tbl[p], n)
                s = -10 * d + 20 * own - 5 * foreign - 5 * blocked
                if s > 1000:
                    s = 1000
                if s < -1000:
                    s = -1000
            pack |= (s + OFF) << (15 * p)
        return pack

    @njit(cache=True)
    def maxn_core(board, player, depth, b0, b1, b2, b3, tt, nodes, goal_tbl):
        # same search as maxn(), values packed into one int64
        nodes[0] += 1

        ds = np.empty(25, dtype=np.int64)
        for i in range(25):
            ds[i] = (board // pow5_tbl[i]) % 5

        pieces = np.zeros((4, 3), dtype=np.int64)
        counts = np.zeros(4, dtype=np.int64)
        for i in range(25):
            d = ds[i]
            if d >= 1:
                pieces[d - 1, counts[d - 1]] = i
                counts[d - 1] += 1

        w = 0
        for p in range(4):
            if counts[p] == 3:
                allin = True
                for k in range(3):
                    cell = pieces[p, k]
                    if not (cell == goal_tbl[p, 0] or cell == goal_tbl[p, 1]
                            or cell == goal_tbl[p, 2]):
                        allin = False
                if allin:
                    w = p + 1
        if w != 0:
            pack = 0
            for p in range(4):
                val = 10000 + depth if p + 1 == w else -10000 - depth
                pack |= (val + OFF) << (15 * p)
            return pack, -1, -1
        if depth == 0:
            return eval_core(pieces, counts, goal_tbl), -1, -1

        ti = board % tt.shape[0]
        e = tt[ti]
        if e[0] == board and e[1] == player and e[3] >= depth:
            return e[2], e[4], e[5]

        pi = player - 1
        f_arr = np.empty(24, dtype=np.int64)
        t_arr = np.empty(24, dtype=np.int64)
        n_moves = 0
        for k in range(counts[pi]):
            i = pieces[pi, k]
            for s in range(4):
                tgt = step_tbl[i, s]
                if tgt >= 0 and ds[tgt] == 0:
                    f_arr[n_moves] = i
                    t_arr[n_moves] = tgt
                    n_moves += 1
            for s in range(4):
                mid = jump_tbl[i, s, 0]
                land = jump_tbl[i, s, 1]
                if mid >= 0 and ds[mid] != 0 and ds[land] == 0:
                    f_arr[n_moves] = i
                    t_arr[n_moves] = land
                    n_moves += 1

        if n_moves == 0:
            return eval_core(pieces, counts, goal_tbl), -1, -1

        cur = assign_core(pieces[pi], goal_tbl[pi], counts[pi])
        keys = np.empty(n_moves, dtype=np.int64)
        np3 = np.empty(3, dtype=np.int64)
        for m in range(n_moves):
            f = f_arr[m]
            t = t_arr[m]
            for k in range(counts[pi]):
                np3[k] = t if pieces[pi, k] == f else pieces[pi, k]
            gain = cur - assign_core(np3, goal_tbl[pi], counts[pi])
            jump = 1 if manh_tbl[f, t] == 2 else 0
            own = 1 if (t == goal_tbl[pi, 0] or t == goal_tbl[pi, 1]
                        or t == goal_tbl[pi, 2]) else 0
            foreign = 0
            for q in range(4):
                if q != pi and (t == goal_tbl[q, 0] or t == goal_tbl[q, 1]
                                or t == goal_tbl[q, 2]):
                    foreign = 1
            keys[m] = (((64 - gain) << 13) | ((1 - jump) << 12)
                       | ((1 - own) << 11) | (foreign << 10)
                       | ((24 - f) << 5) | (24 - t))
        if e[0] == board and e[1] == player and e[4] >= 0:
            for m in range(n_moves):
                if f_arr[m] == e[4] and t_arr[m] == e[5]:
                    keys[m] -= 1 << 28
        order = np.argsort(keys)

        # beam limit, same schedule as the pure python search
        if depth >= 9:
            n_iter = min(n_moves, limits_tbl[0])
        else:
            n_iter = min(n_moves, limits_tbl[8 - depth])

        best_pack = 0
        best_f = -1
        best_t = -1
        best_vpi = -BIG - 1
        complete = True
        inh = b0 if pi == 0 else (b1 if pi == 1 else (b2 if pi == 2 else b3))
        ob0, ob1, ob2, ob3 = b0, b1, b2, b3
        for mm in range(n_iter):
            m = order[mm]
            f = f_arr[m]
            t = t_arr[m]
            nb = board + player * (pow5_tbl[t] - pow5_tbl[f])
            vpack, _, _ = maxn_core(nb, player % 4 + 1, depth - 1,
                                    ob0, ob1, ob2, ob3, tt, nodes, goal_tbl)
            vpi = ((vpack >> (15 * pi)) & MASK) - OFF
            if vpi > best_vpi:
                best_vpi = vpi
                best_pack = vpack
                best_f = f
                best_t = t
                if pi == 0:
                    ob0 = max(ob0, vpi)
                elif pi == 1:
                    ob1 = max(ob1, vpi)
                elif pi == 2:
                    ob2 = max(ob2, vpi)
                else:
                    ob3 = max(ob3, vpi)
                if vpi > inh:
                    complete = False
                    break
        if complete:
            tt[ti, 0] = board
            tt[ti, 1] = player
            tt[ti, 2] = best_pack
            tt[ti, 3] = depth
            tt[ti, 4] = best_f
            tt[ti, 5] = best_t
        return best_pack, best_f, best_t

    tt_size = 2 ** 20
    tt_buf = np.zeros((tt_size, 6), dtype=np.int64)
    nodes_buf = np.zeros(1, dtype=np.int64)
    warm_goal = np.array([goal_lists[p] for p in PLAYERS], dtype=np.int64)

    def make_goal_table():
        return np.array([goal_lists[p] for p in PLAYERS], dtype=np.int64)

    def decide(enc, player, goal_tbl):
        global NODES
        tt_buf[:, :] = 0
        nodes_buf[0] = 0
        pack, f, t = maxn_core(enc, player, SEARCH_DEPTH * 4, BIG, BIG,
                               BIG, BIG, tt_buf, nodes_buf, goal_tbl)
        NODES = int(nodes_buf[0])
        return (f, t) if f >= 0 else None

    # the first call pays the compile time; cache=True makes later starts fast
    print('Warming up AI...')
    maxn_core(0, 1, 0, BIG, BIG, BIG, BIG, tt_buf, nodes_buf, warm_goal)
    print('AI is ready.')
    return decide, make_goal_table


def get_jit():
    global jit_core
    if jit_core is None:
        try:
            jit_core = init_jit()
        except ImportError:
            print('Warning: numba is not installed using the pure python version')
            jit_core = False
    return jit_core or None


def AI_Player_Team29(board, player, flag):
    # the main function the game engine calls
    enc = validate(board, player)
    if enc is None:
        return fallback_move(board, player)

    set_zones(detect_zones(enc))

    if flag:
        reset_nodes()
        tt = {}
        viz = TreeRecorder(MAX_TREE_NODES)
        value, move = maxn(enc, player, SEARCH_DEPTH * 4, (INF,) * 4, tt,
                           viz, None, 'ROOT P%d' % player)
        if move is None:
            return fallback_move(board, player)
        render_tree(viz)
        return (to_ref(move[0]), to_ref(move[1]))

    core = get_jit()
    if core is None:
        reset_nodes()
        tt = {}
        value, move = maxn(enc, player, SEARCH_DEPTH * 4, (INF,) * 4, tt,
                           None, None, 'ROOT P%d' % player)
        if move is None:
            return fallback_move(board, player)
        return (to_ref(move[0]), to_ref(move[1]))

    decide, make_goal_table = core
    move = decide(enc, player, make_goal_table())
    if move is None:
        return fallback_move(board, player)
    return (to_ref(move[0]), to_ref(move[1]))
