import numpy as np
import random
from tkinter import *
import random
from constants import *
from formating import *
from cnn import *
global log
global moves

moves = []

def grid_to_moves(grid):
    moves = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] != 0:
                moves.append((i + 1, j + 1))
    return moves

def random_policy():
    move = random.choice(not_used)
    return move

def format_move(old_move):
    y, x = let_to_dig[old_move[0]], int(old_move[1:])
    new_move = (x, y)
    return new_move


def format_from_move(move):
    res = ''
    res += dig_to_let[move[1]]
    res += str(move[0])
    return res


def get_moves_upd(log):
    if log[-1] == '\n':
        log = log[:-1]
    log = log.split()
    for i in range(len(log)):
        log[i] = format_move(log[i])
    return log

def get_grid(log):
    moves = get_moves_upd(log)
    print(moves)
    grid_black = torch.zeros((n + 1, n + 1), dtype=torch.int8)
    grid_white = torch.zeros((n + 1, n + 1), dtype=torch.int8)
    grid_player = torch.ones((n + 1, n + 1), dtype=torch.int8)
    grid_player *= -1
    mvs_num = len(moves)
    print("mvs_num:", mvs_num)
    for idx in range(0, mvs_num - 1, 2):
            grid_black[moves[idx]] = 1
            grid_white[moves[idx + 1]] = -1
    if mvs_num % 2 != 0:
        grid_player *= -1
        grid_black[moves[mvs_num - 1]] = 1
    t = torch.stack([grid_black, grid_white, grid_player])
    t = t.float()
    return t


def lbl_to_move(lbl):
    lbl = int(lbl)
    return ((lbl - 1) // n + 1, (lbl - 1) % n + 1)


def dummy(grid):
    out = model(grid).data.tolist()
    new_out = np.argsort(out).flat
    i = len(new_out) - 1
    while True:
        lbl = new_out[i] + 1
        move = lbl_to_move(lbl)
        if grid[0, move[0], move[1]] == 0 and grid[1, move[0], move[1]] == 0:
            break
        i -= 1
    return move

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


MAX_DEPTH = 5


def minimax(grid, depth, player, x, y, alpha, beta):
    value, _ = is_winning_move(x, y, grid)
    if value:
        return value - 1 * player
    if np.count_nonzero(grid) == n * n or depth == MAX_DEPTH: #no blank cells left
        return 0
    best = -player * 1000
    min_x, min_y, max_x, max_y = find_limits(grid, depth)
    moves = [(x, y) for x in range(min_x, max_x) for y in range(min_y, max_y)]
    moves = sorted(moves, key=lambda a: ((a[0] - (max_x - min_x) // 2)**2 + (a[1] - (max_y - min_y) // 2)**2))
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

def find_best_move(grid, player, moves):
    alpha = -5000
    beta = 5000
    best = -player * 1000
    x, y = -1, -1
    for move in moves:
        i, j = move
        if not (0 < i <= n) or not (0 < j <= n):
            continue
        if grid[i, j] == 0:
            grid[i, j] = player
            move_val = minimax(grid, 1, -player, i, j, alpha, beta)
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
    return (x, y),  np.count_nonzero(grid) == n * n




def open_three(x, y, grid):
    match = [[], [], [], []]
    ends = [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]
    directions = [1, 1, 1, 1]  # ns, we, nw, ne
    parameters = [(1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1), (1, 1, 2), \
                  (-1, -1, 2), (1, -1, 3), (-1, 1, 3)]
    for p in parameters:
        in_boundaries(grid, x, y, p, directions, match)
    streak = max(directions)
    print("streak: ", streak)
    for d in range(len(directions)):
        if directions[d] >= goal - 2:
            moves = match[d]
            moves.append((x, y))
            moves.sort()
            print(moves)
            open = 0
            x_b, y_b = moves[-1][0] + parameters[2 * d][0], moves[-1][1] + parameters[2 * d][1]
            x_e, y_e = moves[0][0] + parameters[2 * d + 1][0], moves[0][1] + parameters[2 * d + 1][1]
            if 0 < x_b <= n and 0 < y_b <= n and not grid[x_b, y_b]:
                open += 1
                ends[d][0] = (x_b, y_b)
            if 0 < x_e <= n and 0 < y_e <= n and not grid[x_e, y_e]:
                open += 1
                ends[d][1] = (x_e, y_e)
            print("moves: ", moves, "(x_b, y_b): ", x_b, y_b, "(x_e, y_e)", x_e, y_e, "d: ", d)
            print("open: ", open)
    return ends



def make_move(log):
    st = time.time()
    log.strip()
    grid = get_grid(log)
    moves = get_moves_upd(log)
    with torch.no_grad():
        move_net = dummy(grid)
    flat_grid = grid[0, :, :] + grid[1, :, :]
    last_x, last_y = moves[-1]
    player = int(flat_grid[last_x, last_y])
    if player == 1:
        plr_moves = moves[0:: 2]
    else:
        plr_moves = moves[1:: 2]
    ends = open_three(last_x, last_y, flat_grid)
    if ends != [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]:
        positions = [p for pos in ends for p in pos if p != (-1, -1)]
        if move_net in positions:
            move_defence = move_net
        else:
            move_defence = random.choice(positions)
        mv = move_defence
    else:
        move_attack = move_net
        for move in plr_moves:
            if time.time() - st > 2.6:
                break
            ends = open_three(move[0], move[1], flat_grid)
            if ends != [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]:
                positions = [p for pos in ends for p in pos if p != (-1, -1)]
                if move_net in positions:
                    break
                move_attack = random.choice(positions)
        mv = move_attack
    return mv


def get_moves_upd(log):
    if log[-1] == '\n':
        log = log[:-1]
    log = log.split()
    for i in range(len(log)):
        log[i] = format_move(log[i])
    return log


clicked = np.zeros((n, n), dtype=np.int8)  # Создаем сет для клеточек, по которым мы кликнули
not_used = list(range(1, n * n + 1))

policy = dummy

global prev_x, prev_y
prev_x, prev_y = -1, -1

def click(event):
    global prev_x, prev_y
    global moves
    ids = c.find_withtag(CURRENT)[0]  # Определяем по какой клетке кликнули
    print(ids)
    x = (ids - 1) % n
    y = (ids - 1) // n
    clicked[x][y] = 1
    moves.append((x + 1, y + 1))
    not_used.remove(ids)
    c.itemconfig(CURRENT, fill="#C7007D")
    log = get_log(moves, '')
    x, y = make_move(log)
    moves.append((x, y))
    x -= 1
    y -= 1
    ids = y * n + x + 1
    print("(x, y):", x, y)
    print(ids)
    not_used.remove(ids)
    clicked[x][y] = -1
    c.itemconfig(ids, fill="#1CA9C9")
    c.update()
    game_is_over = game_over()
    if (game_is_over != 0):
        print("GAME OVER, WINNER IS ", game_is_over)


def who_walks():
    return step % 2


def game_over():
    b = 0
    w = 0
    for i in range(15):  # по строкам
        w = 0
        b = 0
        for j in range(15):
            if (clicked[i][j]) == 1:
                b += 1
                w = 0
            elif clicked[i][j] == -1:
                b = 0
                w += 1
            else:
                b = 0
                w = 0
            if b >= 5:
                return 1
            if w >= 5:
                return -1

    for j in range(15):  # по столбцам
        w = 0
        b = 0
        for i in range(15):
            if (clicked[i][j]) == 1:
                b += 1
                w = 0
            elif clicked[i][j] == -1:
                b = 0
                w += 1
            else:
                b = 0
                w = 0
            if b >= 5:
                return 1
            if w >= 5:
                return -1

    for i in range(11):
        w = 0
        b = 0
        for k in range(15 - i):
            if (clicked[i + k][k]) == 1:
                b += 1
                w = 0
            elif clicked[i + k][k] == -1:
                b = 0
                w += 1
            else:
                b = 0
                w = 0

            if b >= 5:
                return 1
            if w >= 5:
                return -1

    for i in range(n - goal + 1):
        w = 0
        b = 0
        for k in range(15 - i):
            if (clicked[k][i + k]) == 1:
                b += 1
                w = 0
            elif clicked[k][i + k] == -1:
                b = 0
                w += 1
            else:
                b = 0
                w = 0
            if b >= 5:
                return 1
            if w >= 5:
                return -1

    return 0


def can_use(i, j):
    if clicked[i][j] == 0:
        return 1
    return 0


GRID_SIZE = 15  # Ширина и высота игрового поля
SQUARE_SIZE = 30  # Размер одной клетки на поле
global step
step = 0

root = Tk()  # Основное окно программы
root.title("連珠")
c = Canvas(root, width=GRID_SIZE * SQUARE_SIZE,
           height=GRID_SIZE * SQUARE_SIZE)  # Задаем область на которой будем рисовать
c.pack()

c.bind("<Button-1>", click)
# Следующий код отрисует решетку из клеточек серого цвета на игровом поле
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        c.create_rectangle(i * SQUARE_SIZE, j * SQUARE_SIZE,
                           i * SQUARE_SIZE + SQUARE_SIZE,
                           j * SQUARE_SIZE + SQUARE_SIZE, fill='#FFCBDB')

root.mainloop()  # Запускаем программу