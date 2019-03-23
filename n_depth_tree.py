from constants import *
from formating import *

def in_boundaries(grid, x, y, parameters, directions, match):
    x_idx, y_idx, direction = parameters
    for i in range(1, goal):
        if x + i * x_idx <= 0 or x + i * x_idx > n or y + i * y_idx <= 0 \
                or y + i * y_idx > n or grid[x + i * x_idx, y + i * y_idx] != grid[x, y]:
            break
        directions[direction] += 1
        match[direction].append((x + i * x_idx, y + i * y_idx))


def intersects(pos, x1, y1, x2, y2):
    x, y = pos
    return x1 <= x <= x2 and y1 <= y <= y2


def is_winning_move(x, y, grid):
    match = [[], [], [], []]
    moves = []
    value = 0  # 0 when it's a non-winning move, 100 if 1-st player wins and -100 if 2-d player wins
    directions = [1, 1, 1, 1]  # ns, we, nw, ne
    parameters = [(1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1), (1, 1, 2), \
                  (-1, -1, 2), (1, -1, 3), (-1, 1, 3)]
    for p in parameters:
        in_boundaries(grid, x, y, p, directions, match)
    is_winning = (max(directions) >= goal)
    if is_winning:
        moves = match[np.argmax(directions)]
        moves.append((x, y))
        value = grid[x, y] * 100
    return value, moves


MAX_DEPTH = 3


def minimax(grid, depth, player, x, y, alpha, beta):
    #print()
    #print(grid)
    #print()
    value, _ = is_winning_move(x, y, grid)
    if value:
        return value - 1 * player
    if np.count_nonzero(grid) == n * n or depth == MAX_DEPTH: #no blank cells left
        return 0
    best = -player * 1000
    min_x, min_y, max_x, max_y = find_limits(grid, depth)
    moves = [(x, y) for x in range(min_x, max_x) for y in range(min_y, max_y)]
    moves = sorted(moves, key=lambda a: ((a[0] - (max_x - min_x) // 2)**2 + (a[1] - (max_y - min_y) // 2)**2))
    #print(moves)
    for move in moves:
        i, j = move
        if grid[i, j] == 0:
            grid[i, j] = player
            val = minimax(grid, depth + 1, -player, i, j, alpha, beta)
            if player == 1:
                best = max(val, best)
                alpha = max(alpha, best)
            else:
                best = min(val, best)
                beta = min(beta, best)
            grid[i, j] = 0
        if beta <= alpha:
            break
    return best - 1 * player


def get_best(player, a, b):
    if player == 1:
        return a < b
    return a > b

def find_limits(grid, depth):
    min_x, min_y = n + 1, n + 1
    max_x, max_y = 0, 0
    if np.count_nonzero(grid) == 0:
        return 1, 1, n + 1, n + 1
    for i in range(n):
        for j in range(n):
            if grid[i, j] != 0:
                max_x = max(max_x, i)
                max_y = max(max_y, j)
                min_x = min(min_x, i)
                min_y = min(min_y, j)
    min_x = max(1, min_x - (MAX_DEPTH - depth) // 2)
    min_y = max(1, min_y - (MAX_DEPTH - depth) // 2)
    max_x = min(n + 1, max_x + (MAX_DEPTH - depth) // 2 + 1)
    max_y = min(n + 1, max_y + (MAX_DEPTH - depth) // 2 + 1)
    return min_x, min_y, max_x, max_y

def find_best_move(grid, player):
    #print("___________________________________")
    alpha = -5000
    beta = 5000
    best = -player * 1000
    x, y = -1, -1
    min_x, min_y, max_x, max_y = find_limits(grid, 0)
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            if grid[i, j] == 0:
                grid[i, j] = player
                move_val = minimax(grid, 1, -player, i, j, alpha, beta)
                #print(grid)
                #print(move_val)
                if get_best(player, best, move_val):
                    best = move_val
                    x = i
                    y = j
                    value, _ = is_winning_move(x, y, grid)
                    if value:
                        return (x, y), True
                grid[i, j] = 0
    while x == -1 and y == -1 and np.count_nonzero(grid) != n * n:
        t_x = np.random.randint(n)
        t_y = np.random.randint(n)
        if grid[t_x][t_y] == 0:
            x = t_x
            y = t_y
            break
    #print("___________________________________")
    return (x, y),  np.count_nonzero(grid) == n * n


grid = np.zeros((n + 1, n + 1), dtype=np.int8)
#winner, moves = get_moves("white h8 j9 k7 j7 j8 h9 k8 l8 k9 k6 l10 m11 h10 k10 f10 f8 e9 e10 d8 c7 d9 f7 d7 d10 c9 f9 f6 e7 d6 d5 g8 g6 g7 h5 j4 e5 f5 e6 e4 c4 b3 m12 l11 l12 k12 k13 n10 m13")
winner, moves = get_moves("white b4 c4 a5 c5 a3 c3")
print(flip_log("white b4 c4 a5 c5 a3 c3"))
#TEST THIS ONE

#winner, moves = get_moves("white a1 c2 b3")
#moves = []
grid = moves_to_grid(moves)
print(grid)
print(moves)
player = 1
game_over = False
last_move = (0, 0)
while not game_over:
    move, game_over = find_best_move(grid, player)
    grid[move] = player
    print(move, player)
    player = -player
    print(grid)
    last_move = move
if last_move != (-1, -1):
    print("Player ", -player, " wins")
else:
    print("It's a draw")
