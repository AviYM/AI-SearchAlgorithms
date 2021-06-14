# 203043401 Avi Miletzky

import heapq
import math
import random

# the func returns the revers of a fiven collection.
def reverse_set(coll):
    return [s for s in reversed(coll)]


# ********************************* The PriorityQueue Class *********************************#

# running a queue with priority ordering which enable to insert and pop items at a certain order
# the priority defined by function f such that the pop function return the highest priority node
class PriorityQueue:

    # constructor function that initialize the inner members of the class.
    def __init__(self, f=lambda x: x):
        self.heap = []
        self.f = f

    # the func enters the given item to the queue so that its position determined by its priority = f(item).
    def append(self, item, priority=0):
        heapq.heappush(self.heap, ((self.f(item), priority), item))

    # insert list of items to the queue
    def extend(self, items):
        for item in items:
            self.append(item)

    # returns the item with the highest priority that is located in index 1.
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    # returns the queue size
    def __len__(self):
        return len(self.heap)

    # check if queue contains item with such key
    def __contains__(self, key):
        return any([item == key for _, item in self.heap])

    # returns item by the key
    def __getitem__(self, key):
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    # delete item with such key
    def __delitem__(self, key):
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

    # string format of the heap
    def __repr__(self):
        return str(self.heap)


# ********************************* The Node Class *********************************#

# node class describe a single node of the problem, holding necessary data needed to solve the problem
# the node data is comprised of several parameters such as its state, its parent and so on
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    # returns all the child nodes that are the optional states by moving every direction from the current state.
    def expand(self, problem, reverse_flag=0):
        actions = problem.actions(self.state)
        if reverse_flag:
            actions = reverse_set(actions)
        return [self.child_node(problem, action)
                for action in actions]

    # returns the child node of moving from current state by the parameter action.
    def child_node(self, problem, action):
        next_state = problem.succ(self.state, action)
        next_node = Node(next_state, self, action,
                         self.path_cost + problem.step_cost(self.state, action))
        return next_node

    # returns the solution, namely the moves list that required to get to this node.
    def solution(self):
        return '-'.join([node.action for node in self.path()[1:]])

    # returns the path, namely nodes list that are on the way to this node.
    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # returns state in string format.
    def __repr__(self):
        return f"<{self.state}>"

    # 'less than' operation implemented by comparing nodes state.
    def __lt__(self, node):
        return self.state < node.state

    # 'equal' operation implemented by comparing nodes instances and state.
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    # 'not equal' operation implemented by negation of equal operation
    def __ne__(self, other):
        return not (self == other)

    # def __hash__(self):
    #    return hash(self.state)


# ********************************* The Problem Class *********************************#

# this class defines the our problem and contains G-graph size n*n and two points s_start and goal.
# the algorithms to which we send the problem will need to find a path from s_start to the goal
# by keeping legal movements as defined in moves.
class RoutingProblem:

    # constructor function that initialize the inner members of the class.
    def __init__(self, s_start, goal, n, G):
        self.s_start = s_start
        self.goal = goal
        self.n = n
        self.G = G
        self.moves = {
            'R': lambda x: ((x[0] < n - 1) and (self.G[x[1]][(x[0] + 1)] != -1)),
            'RD': lambda x: ((x[0] < n - 1) and (x[1] < n - 1) and (self.G[(x[1] + 1)][(x[0] + 1)] != -1)
                             and (self.G[x[1]][(x[0] + 1)] != -1) and (self.G[(x[1] + 1)][x[0]] != -1)),
            'D': lambda x: ((x[1] < n - 1) and (self.G[(x[1] + 1)][x[0]] != -1)),
            'LD': lambda x: ((x[0] > 0) and (x[1] < n - 1) and (self.G[(x[1] + 1)][(x[0] - 1)] != -1)
                             and (self.G[x[1]][(x[0] - 1)] != -1) and (self.G[(x[1] + 1)][x[0]] != -1)),
            'L': lambda x: ((x[0] > 0) and (self.G[x[1]][(x[0] - 1)] != -1)),
            'LU': lambda x: ((x[0] > 0) and (x[1] > 0) and (self.G[(x[1] - 1)][(x[0] - 1)] != -1)
                             and (self.G[x[1]][(x[0] - 1)] != -1) and (self.G[(x[1] - 1)][x[0]] != -1)),
            'U': lambda x: ((x[1] > 0) and (self.G[(x[1] - 1)][x[0]] != -1)),
            'RU': lambda x: ((x[0] < n - 1) and (x[1] > 0) and (self.G[(x[1] - 1)][(x[0] + 1)] != -1)
                             and (self.G[x[1]][(x[0] + 1)] != -1) and (self.G[(x[1] - 1)][x[0]] != -1))
        }
        self.transitions = {
            'R': (1, 0),
            'RD': (1, 1),
            'D': (0, 1),
            'LD': (-1, 1),
            'L': (-1, 0),
            'LU': (-1, -1),
            'U': (0, -1),
            'RU': (1, -1)
        }

    # the func returns a list of the legal actions for the given state s.
    def actions(self, s):
        return [k for (k, f) in self.moves.items() if f(s)]

    # the func returns a new state for action a in the current state(s).
    def succ(self, s, a):
        d = self.transitions[a]
        return (s[0] + d[0], s[1] + d[1])

    # the func returns true if the given state(s) = goal, and false otherwise.
    def is_goal(self, s):
        return s == self.goal

    # the func returns the step cost for action a in the current state(s).
    def step_cost(self, s, a):
        d = self.transitions[a]
        return self.G[(s[1] + d[1])][(s[0] + d[0])]

    # returns the state as string representation.
    def state_str(self, s):
        return ','.join([str(i) for i in s])


# ********************************* IDS Algorithm *********************************#

# returns the solution only if goal is reachable from s_start within 'limit' steps.
def depth_limited_search(problem, limit):
    expanded_counter = 0
    frontier = [(Node(problem.s_start))]  # Stack
    while frontier:
        node = frontier.pop()
        # print(f'The node is:  {node}\t The frontier is: {frontier}')
        if problem.is_goal(node.state):
            return (' '.join([node.solution(), str(node.path_cost)])), expanded_counter
        if node.depth < limit:
            frontier.extend(node.expand(problem, 1))
        expanded_counter += 1
    return None, expanded_counter


# the func runs repeatedly with increasing depth limits until the goal is found, in such case it returns the solution.
# or until the depth >= max_depth and returns None.
def iterative_deepening_search(problem, max_depth=20):
    all_expanded_counter = 0
    for depth in range(0, max_depth):
        solution, expanded_cntr = depth_limited_search(problem, depth)
        all_expanded_counter += expanded_cntr
        # print(depth,'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if solution:
            return ' '.join([solution, str(all_expanded_counter)])
    return None


# ********************************* Best First Graph Search Algorithm *********************************#

# this algorithm searches a path from s_start to the goal,
# by expanding the most promising node chosen according to a specified given rule f.
def best_first_graph_search(problem, f):
    creat_order = 0;
    expanded_counter = 0
    node = Node(problem.s_start)
    frontier = PriorityQueue(f)
    frontier.append(node)
    closed_list = set()
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return (' '.join([node.solution(), str(node.path_cost), str(expanded_counter)]))
        expanded_counter += 1
        closed_list.add(node.state)
        for child in node.expand(problem):
            creat_order += 1
            if child.state not in closed_list and child not in frontier:
                frontier.append(child, creat_order)
            elif child in frontier and f(child) < frontier[child][0]:
                c_o = frontier[child][1]
                del frontier[child]
                frontier.append(child, c_o)
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print(frontier)
        # print(closed_list)
        # expanded_counter += 1
    return None


# returns the accumulated path cost of the given node.
def g(node):
    return node.path_cost


# this algorithm is a variant of Dijikstra and runs the best_first_graph_search by using f = g,
# and therefore it belongs to the family of unknown algorithms.
def uniform_cost_search(problem):
    return best_first_graph_search(problem, f=g)


# ********************************* A* Algorithm *********************************#

# returns the absolute difference between two states.
# consist of the absolute difference on the X-axis = |dx| and the absolute difference on the Y-axis = |dy|.
def abs_distance(state1, state2):
    return abs(state1[0] - state2[0]), abs(state1[1] - state2[1])


# returns the Manhattan distance = |dx| + |dy|
def manhattan_distance(curr_state, goal):
    dx, dy = abs_distance(curr_state, goal)
    return dx + dy


# returns Euclidean_distance = Square root of (dx^2 + dy^2)
def euclidean_distance(curr_state, goal):
    dx, dy = abs_distance(curr_state, goal)
    return int(math.sqrt(((dx ** 2) + (dy ** 2))))


# when D = D2 = 1(the default) this is called the Chebyshev distance and returns the max(dx, dy).
# and when D = 1 and D2 = sqrt(2), this is called the octile distance.
def diagonal_distance(curr_state, goal, D=1, D2=1):
    dx, dy = abs_distance(curr_state, goal)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


# optional admissible heuristic that always <= h*(n), but too far from it.
def optional_heuristic(curr_state, goal):
    return (manhattan_distance(curr_state, goal) - euclidean_distance(curr_state, goal))


# this heuristic func returns the max(abs_distance(node.state, problem.goal)),
# the most minimal steps number between this node to the goal.
def h(node, goal):
    return diagonal_distance(node.state, goal)
    # return euclidean_distance(node.state, problem.goal)
    # return manhattan_distance(node.state, problem.goal)
    # return optional_heuristic(node.state, problem.goal)


# the main func of A* algo. definds the f(n) func of the Best First Graph Search algo to be
# the sum of heuristic func h(n) and the accumulated path cost func g(n). and runs it.
def astar_search(problem):
    return best_first_graph_search(problem, f=lambda n: g(n) + h(n, problem.goal))


# ********************************* IDA* Algorithm *********************************#

# The function checks if there is a circle on the map that includes the given node.
def is_circle(node):
    for n in node.path()[:-1]:
        if n == node:
            return True
    return False


# returns the solution only if goal is reachable from node and the heuristic of each node <= 'f_limit'.
def DFS_Countour(problem, node, f_limit, f, max_depth=20):
    expanded_counter = 0

    if f(node) > f_limit:
        return None, f(node), expanded_counter
    if problem.is_goal(node.state):
        return (' '.join([node.solution(), str(node.path_cost)])), f_limit, expanded_counter

    next_f = math.inf
    expanded_counter += 1
    for child in node.expand(problem):
        if child.depth < max_depth and not is_circle(child):
            solution, new_f, expanded_cntr = DFS_Countour(problem, child, f_limit, f)
            expanded_counter += expanded_cntr
            if solution:
                return solution, f_limit, expanded_counter
            next_f = min(next_f, new_f)
        elif child.depth >= max_depth:
            return None, next_f, expanded_counter

    return None, next_f, expanded_counter


# the func runs repeatedly the DFS_Countour and increasing f_limits until the goal is found.
# in each iteration we will chose the lowest f(node) > f_limit to be the next_f.
def idastar_search(problem):
    all_expanded_counter = 0
    root = Node(problem.s_start)
    f_limit = diagonal_distance(root.state, problem.goal)

    while f_limit != math.inf:
        solution, f_limit, expanded_cntr = DFS_Countour(problem, root, f_limit, f=lambda n: g(n) + h(n, problem.goal))
        all_expanded_counter += expanded_cntr
        if solution:
            return ' '.join([solution, str(all_expanded_counter)])

    return None


# ********************************* Main Function *********************************#

# returns a pointer to the specific algorithm according to the given algo_name string.
def get_algo(algo_name):
    switch = {
        'IDS': iterative_deepening_search,
        'UCS': uniform_cost_search,
        'ASTAR': astar_search,
        'IDASTAR': idastar_search
    }
    return switch.get(algo_name, "Invalid name")


def maps_generator(algo_name, s_start, goal, size, file_name):
    f = open(file_name, "w")  # f'input{(i+10)}.txt'
    f.write(f'{algo_name}\n{s_start}\n{goal}\n{size}')
    for i in range(10):  # rows
        arr = []
        for j in range(10):  # columns
            n = random.randrange(0, 10)
            if n == 0:
                n = -1
            arr.append(n)
        if i == 0:
            arr[0] = 1
        elif i == 9:
            arr[9] = 1
        f.write('\n' + ','.join(str(k) for k in arr))
    f.close()

# the main func reads the input file with the problem, and runs the requested function on the given input
# depending on the contents of the file. then write the return solution to the output file.
def main_func(input_f_name, output_f_name):
    input_f = open(input_f_name, "r")

    algo = input_f.readline().rstrip()
    s_start = tuple([int(i) for i in reversed(input_f.readline().rstrip().split(','))])
    goal = tuple([int(j) for j in reversed(input_f.readline().rstrip().split(','))])
    n = int(input_f.readline())

    graph = []
    for line in input_f:
        graph.append(list([int(i) for i in line.split(',')]))
    # print('\n'.join([str(c) for c in graph]))

    input_f.close()

    rp = RoutingProblem(s_start, goal, n, graph)
    result = get_algo(algo)(rp)

    output_f = open(output_f_name, "a")  # "w"
    output_f.write(input_f_name + '\t')

    if result:
        output_f.write(result)
    else:
        output_f.write('no path')

    output_f.write("\n")
    output_f.close()


#main("input.txt", "output.txt")
# main("input2.txt", "output.txt")
# main("input3.txt", "output.txt")
# main("input4.txt", "output.txt")
for i in range(10):
   file_name = 'input' + str((10 + i)) + '.txt'
   maps_generator('ASTAR','0,0','9,9','10',file_name)
   main_func(file_name, "output.txt")
