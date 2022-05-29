from ctypes import sizeof
from typing import *
import kaggle_environments.helpers as helper
import numpy as np
from copy import deepcopy

class Point(helper.Point):
    def direction_to(self, point: 'Point') -> helper.Direction:
        move_vector = point - self

        if move_vector == helper.Direction.NORTH.to_point():
            return helper.Direction.NORTH

        if move_vector == helper.Direction.SOUTH.to_point():
            return helper.Direction.SOUTH

        if move_vector == helper.Direction.EAST.to_point():
            return helper.Direction.EAST

        if move_vector == helper.Direction.WEST.to_point():
            return helper.Direction.WEST

    def map(self, f: Callable[[int], int]) -> 'Point':
        return Point(f(self[0]), f(self[1]))

    def map2(self, other: Union[Tuple[int, int], 'Point'], f: Callable[[int, int], int]) -> 'Point':
        return Point(f(self[0], other[0]), f(self[1], other[1]))

class Walk:
    def __init__(self) -> None:
        self._points = list()
        self._flight_plan = ''
        self._sub_flight_plans = list() # example : [['N', 1], ['E', 3], ['S', 1]]
        self._valid = True

    def _add_first_step(self, point: Point) -> None:
        if len(self._points) != 0 and point == self._points[0]:
            self._valid = False

        self._points = [point] + self._points
        
        if len(self._points) >= 2:
            first_dir = point.direction_to(self._points[1]).to_char()

            if len(self._sub_flight_plans) == 0:
                self._sub_flight_plans = [[first_dir, 1]]
            elif first_dir == self._sub_flight_plans[0][0]:
                self._sub_flight_plans[0][1] += 1
            else:
                self._sub_flight_plans = [[first_dir, 1]] + self._sub_flight_plans

    def _add_step(self, point: Point) -> None:
        if len(self._points) != 0 and point == self._points[-1]:
            self._valid = False

        self._points += [point]

        if len(self._points) >= 2:
            last_dir = self._points[-2].direction_to(point).to_char()

            if len(self._sub_flight_plans) == 0:
                self._sub_flight_plans = [[last_dir, 1]]
            elif last_dir == self._sub_flight_plans[-1][0]:
                self._sub_flight_plans[-1][1] += 1
            else:
                self._sub_flight_plans += [[last_dir, 1]]

        
    def add_step(self, point: Point, first_step=False) -> bool:
        if first_step:
            self._add_first_step(point)
        else:
            self._add_step(point)

        if self._valid:
            self._flight_plan = self._canonical_flight_plan()
        else:
            self._flight_plan = ''

        return self._valid

    def add_steps(self, points: List[Point]) -> bool:
        valid = True

        for p in points:
            valid &= self.add_step(p)

        return valid

    def _canonical_flight_plan(self):
        fp_str = ''

        if len(self._points) < 2:
            return fp_str

        for d, c in self._sub_flight_plans[:-1]:
            suffix = str(c - 1) if c > 1 else ''
            fp_str += d + suffix

        fp_str += self._sub_flight_plans[-1][0] # No need for the integer on the last straight line

        return fp_str

    @property
    def flight_plan_len(self) -> int:
        return len(self._flight_plan)

    @property
    def points(self) -> List[Point]:
        return self._points

    @property
    def flight_plan(self) -> str:
        return self._flight_plan

    @property
    def valid(self) -> bool:
        return self._valid

    def __len__(self):
        return len(self._points) - 1

    def __str__(self):
        return str(self._points)

    def __repr__(self):
        return str(self._points)

    def copy(self) -> 'Walk':
        c = Walk()
        c._points = self._points.copy()
        c._flight_plan = self._flight_plan
        c._sub_flight_plans = deepcopy(self._sub_flight_plans)
        c._valid = self._valid

        return c
        
class WalkNode:
    def __init__(self, point: Point, parent: 'WalkNode', depth: int, kore: int) -> None:
        self.point = point
        self.kore = kore
        self.parent = parent
        self.children = list()
        self.depth = depth

    def is_leaf(self):
        return len(self.children) == 0

class WalkTree:
    def __init__(self) -> None:
        self._nodes = None
        self._root_idx = 0

    def add_walk(self, walk: Walk) -> None:
        curr_node_idx = self._root_idx

        if self._nodes is None:
            self._nodes = [WalkNode(walk.points[0], None, 0, 0)]

        for depth, p in enumerate(walk.points[1:]):
            exists = False

            for child_idx in self._nodes[curr_node_idx].children:
                if self._nodes[child_idx].point == p:
                    curr_node_idx = child_idx
                    exists = True
                    break

            if not exists:
                new_node = WalkNode(p, curr_node_idx, depth + 1, 0)
                new_node_idx = len(self._nodes)
                self._nodes += [new_node]
                self._nodes[curr_node_idx].children += [new_node_idx]
                curr_node_idx = new_node_idx

    def add_walks(self, walks: List[Walk]) -> None:
        for w in walks:
            self.add_walk(w)
    
    def _recursively_set_kore(self, node_idx: int, kore_map: np.ndarray) -> None:
        x, y = self._nodes[node_idx].point.x, self._nodes[node_idx].point.y
        parent_idx = self._nodes[node_idx].parent

        if parent_idx is not None:
            parent_kore = self._nodes[parent_idx].kore
        else:
            parent_kore = 0

        self._nodes[node_idx].kore = kore_map[x, y] + parent_kore

        for child_idx in self._nodes[node_idx].children:
            self._recursively_set_kore(child_idx, kore_map)

    def set_kore(self, kore_map: np.ndarray) -> None:
        self._recursively_set_kore(0, kore_map)

    def __iter__(self):
        return self._walks_generator()

    def _walks_generator(self):
        # Stack that is used to build walks
        walk_stack = list()
        # Stack of indices to visit
        indices_stack = [self._root_idx]
        prev_idx = None

        while len(indices_stack) != 0:
            # Take the first indice of the stack
            idx = indices_stack[0]
            del indices_stack[0]

            # Compute the delta of depth we've done in the tree compared to the last index
            if prev_idx is not None:
                curr_depth = self._nodes[idx].depth
                prev_depth = self._nodes[prev_idx].depth
                delta = curr_depth - prev_depth
            else:
                delta = 0

            # if we got up in the tree, we need to unstack some walk nodes
            if delta < 0:
                for _ in range(abs(delta) + 1):
                    walk_stack.pop()

            # add the current node to the build stack
            walk_stack += [self._nodes[idx].point]

            # if that node is a leaf we build a walk out of the stack and return it
            if self._nodes[idx].is_leaf():
                walk = Walk()
                walk.add_steps(walk_stack)
                yield walk, self._nodes[idx].kore

            # add all children to the visit stack
            for child_idx in self._nodes[idx].children:
                indices_stack = [child_idx] + indices_stack

            prev_idx = idx

    def leaves(self):
        return {idx:self._nodes[idx] for idx in range(len(self._nodes)) if self._nodes[idx].children == []}

            

def max_fp_len(fleet_size):
    return int(np.floor(2 * np.log(fleet_size)) + 1)

if __name__ == "__main__":
    import pickle
    import random

    walks = pickle.load(open("/workspaces/kore/kore-2022/notebooks/all_walks.pkl", "rb"))
    tree = WalkTree()
    walks = list()

    kore_map = np.random.random((20, 20))

    for _ in range(100):
        walk = Walk()
        walk.add_step(Point(10, 10))

        for i in range(10):
            p = helper.Direction.from_index(random.randint(0, 3)).to_point()
            p = Point(p.x, p.y)
            walk.add_step(walk.points[-1] + p)

        walks += [walk]

    tree.add_walks(walks)
    tree.set_kore(kore_map)
    print(tree)
    # pickle.dump(walks, open("list.pkl", "wb"))
    # pickle.dump(tree, open("tree.pkl", "wb"))
