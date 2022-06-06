from copy import deepcopy
from typing import List
import numpy as np
import pickle
from kore.utils import Walk, Point, distance



def fuse_valid_walks(pos: Point, walks: List[Walk], max_fp_len):
    new_walks = []

    for w in walks:
        wc = w.copy()
        valid = wc.add_step(pos, first_step=True)

        if valid and wc.flight_plan_len <= max_fp_len:
            new_walks += [wc]

    return new_walks

# TODO : Symetry can be used to reduce compute time ?
def list_mining_walks(grid_shape: tuple, start: Point, end: Point, max_length=10, max_fp_len=7, first_call=True, _starting_pos=None):
    m, n = grid_shape
    
    if first_call:
        _starting_pos = start
        
    if max_length == 0 or (not first_call and start == end):
        w = Walk()
        w.add_step(start)
        return [w]

    adj_pos = list()
    
    if start[0] - 1 >= 0:
        adj_pos += [Point(start[0] - 1, start[1])]

    if start[0] + 1 <= m - 1:
        adj_pos += [Point(start[0] + 1, start[1])]

    if start[1] - 1 >= 0:
        adj_pos += [Point(start[0], start[1] - 1)]
        
    if start[1] + 1 <= n - 1:
        adj_pos += [Point(start[0], start[1] + 1)]

    walks = list()

    for p in adj_pos:
        if _starting_pos != end:
            no_return = p != _starting_pos
        else:
            no_return = True

        if distance(p, end) <= max_length - 1 and no_return:
            p_walks = list_mining_walks(grid_shape, p, end, max_length - 1, first_call=False, _starting_pos=_starting_pos)
            walks += fuse_valid_walks(start, p_walks, max_fp_len)

    return walks

def mining_rate(fleet_size):
    return np.log(fleet_size) / 20

def mined_kore_on_walk(walk, board, fleet_size):
    board_copy = deepcopy(np.asarray(board, dtype=np.float32))
    mr = mining_rate(fleet_size)
    kore = 0
    
    for p in walk[1:-1]:
        kore_mined = mr * board_copy[p[0], p[1]]
        kore += kore_mined
        board_copy[p[0], p[1]] -= kore_mined
        board_copy += board_copy * 0.02

    return kore

def kore_on_walk(walk, board):
    kore = 0
    for p in walk[1:-1]:
        kore += board[p.x, p.y]

    return kore

def rank_mining_walks(walks, board, fleet_size, pos_to_walk):
    reduced_walks = list()
    walk_indices = list()

    for i, j in zip(range(board.shape[0]), range(board.shape[1])):
        if board[i, j] > 0 and (i, j) in pos_to_walk:
            walk_indices += pos_to_walk[(i, j)]

    walk_indices = list(set(walk_indices))
    reduced_walks = [walks[i] for i in walk_indices]

    ranked_walks = dict()
    for w in reduced_walks:
        ranked_walks[w.flight_plan_len] = ranked_walks.get(w.flight_plan_len, []) + [(w, mined_kore_on_walk(w.points, board, fleet_size) / len(w))]

    for k in ranked_walks:
        ranked_walks[k] = sorted(ranked_walks[k], key=lambda x: x[1])[::-1]

    return ranked_walks

def rank_mining_walks_bf(walks, board):
    ranked_walks = [(w, kore_on_walk(w.points, board) / len(w)) for w in walks]
    ranked_walks = sorted(ranked_walks, key=lambda x: x[1])[::-1]
    ranked_walks = [w[0] for w in ranked_walks]

    return ranked_walks

# @functools.lru_cache
def get_shipyard_mining_walks(shipyard_position, board):
    walks = pickle.load(open("/workspaces/kore/kore-2022/notebooks/all_walks.pkl", "rb"))
    local_board = deepcopy(board)
    x, y = shipyard_position[0], shipyard_position[1]
    x_indices = [(x + offset) % board.shape[0] for offset in range(-5, 6)]
    y_indices = [(y + offset) % board.shape[1] for offset in range(-5, 6)]
    local_board = local_board[x_indices,:]
    local_board = local_board[:,y_indices]

    ranked_walks = rank_mining_walks_bf(walks, local_board)
    ranked_walks_by_size = dict()

    for w in ranked_walks:
        ranked_walks_by_size[w.flight_plan_len] = ranked_walks_by_size.get(w.flight_plan_len, []) + [w]

    return ranked_walks_by_size

if __name__ == "__main__":
    import pickle

    valid_walks = list_mining_walks((11, 11), Point(5, 5), Point(5, 5), max_length=10, max_fp_len=7)
    pickle.dump(valid_walks, open("../notebooks/all_walks.pkl", "wb"))