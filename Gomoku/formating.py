import numpy as np
import torch
from constants import *

let_to_dig = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, \
              'h': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15}

dig_to_let = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', \
              8: 'h', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p'}


def format_move(old_move):
    y, x = let_to_dig[old_move[0]], int(old_move[1:])
    new_move = (x, y)
    return new_move


def format_from_move(move):
    res = ''
    res += dig_to_let[move[1]]
    res += str(move[0])
    return res


def get_moves(log):
    if log[-1] == '\n':
        log = log[:-1]
    log = log.split()
    winner = log[0]
    log = log[1:]
    for i in range(len(log)):
        log[i] = format_move(log[i])
    return winner, log


def get_log(moves, winner):
    log = winner
    for move in moves:
        log += ' '
        log += format_from_move(move)
    log += '\n'
    return log


def moves_to_grid(moves):
    grid = np.zeros((n + 1, n + 1), dtype=np.int8)
    player = 1
    for move in moves:
        grid[move] = player
        player *= -1
    return grid


def move_to_lbl(move):
    return (move[0] - 1) * n + move[1]


def lbl_to_move(lbl):
    lbl = int(lbl)
    return ((lbl - 1) // n + 1, (lbl - 1) % n + 1)


def process_log(log):
    winner, moves = get_moves(log)
    grid_black = torch.zeros((n + 1, n + 1), dtype=torch.int8)
    grid_white = torch.zeros((n + 1, n + 1), dtype=torch.int8)
    grid_player = torch.ones((n + 1, n + 1), dtype=torch.int8)
    grids = []
    labels = []
    if winner == 'white':
        grid_player *= -1
        for idx in range(1, len(moves), 2):
            grid_black[moves[idx - 1]] = 1
            lbl = move_to_lbl(moves[idx])
            t = torch.stack([grid_black, grid_white, grid_player])
            grids.append(t)
            labels.append(lbl)
            grid_white[moves[idx]] = -1
    else:
        grid_black[moves[0]] = 1
        for idx in range(0, len(moves), 2):
            grid_white[moves[idx - 1]] = -1
            lbl = move_to_lbl(moves[idx])
            t = torch.stack([grid_black, grid_white, grid_player])
            grids.append(t)
            labels.append(lbl)
            grid_black[moves[idx]] = 1
    return grids, labels

'''
grids, labels = process_log("black a1 b2 c3") #"white a1 h9 h8 g6 h7 f7 f8 e6 g7 g8 j10 d5")
for i in range(len(labels)):
    print(grids[i][0] + grids[i][1], labels[i] // n, labels[i] % n)
'''


def get_data_n_labels(logs):
    data = []
    labels = []
    for log in logs:
        d, l = process_log(log)
        data += d
        labels += l
    return data, labels


def log_to_data(log):
    shifted_logs = get_shifted_logs(log)
    rotated = get_rotated_logs(log)
    for r in rotated:
        shifted_logs += get_shifted_logs(r)
    flipped = flip_log(log)
    shifted_logs += get_shifted_logs(flipped)
    shifted_logs = list(set(shifted_logs))
    d, l = get_data_n_labels(shifted_logs)
    return d, l



def find_shifts(moves, x, y, dir_x, dir_y, winner):
    new_logs = []
    for y_shift in range(y):
        for x_shift in range(x):
            new_moves = []
            for move in moves:
                new_moves.append((move[0] + dir_x * x_shift, move[1] + dir_y * y_shift))
            new_logs.append(get_log(new_moves, winner))
            #print(new_logs[-1])
    return new_logs


def get_shifted_logs(log):
    new_logs = []
    min_x, min_y, max_x, max_y = n - 1, n - 1, 0, 0
    winner, moves = get_moves(log)
    for move in moves:
        x, y = move
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
    new_logs += find_shifts(moves, min_x, min_y, -1, -1, winner)
    new_logs += find_shifts(moves, min_x, n + 1 - max_y, -1, 1, winner)
    new_logs += find_shifts(moves, n + 1 - max_x, n + 1- max_y, 1, 1, winner)
    new_logs += find_shifts(moves, n + 1 - max_x, min_y, 1, -1, winner)
    return list(set(new_logs))


def flip_log(log):
    winner, moves = get_moves(log)
    flipped = []
    for move in moves:
        flipped.append((move[0], n - move[1] + 1))
    flipped_log = get_log(flipped, winner)
    return flipped_log


def rotate_log(log):
    winner, moves = get_moves(log)
    rotated = []
    for move in moves:
        rotated.append((move[1], n - move[0] + 1))
    rotated_log = get_log(rotated, winner)
    return rotated_log


def get_rotated_logs(log):
    rotated = log
    rotated_logs = []
    for i in range(3):
        print()
        rotated = rotate_log(rotated)
        rotated_logs.append(rotated)
    return rotated_logs


def in_boundaries(grid, x, y, parameters, directions, match, ends=None):
    x_idx, y_idx, direction = parameters
    for i in range(1, goal):
        if x + i * x_idx < 0 or x + i * x_idx >= n or y + i * y_idx < 0 \
                or y + i * y_idx >= n or grid[x + i * x_idx, y + i * y_idx] != grid[x, y]: #TEMPORARY CHANGE OF EQUALITIES
            break

        directions[direction] += 1
        match[direction].append((x + i * x_idx, y + i * y_idx))


def intersects(pos, x1, y1, x2, y2):
    x, y = pos
    return x1 <= x <= x2 and y1 <= y <= y2


def is_winning_move(x, y, grid):
    match = [[], [], [], []]
    moves = []
    value = 0  # 0 when it's a non-winning move, 10 if 1-st player wins and -10 if 2-d player wins
    directions = [1, 1, 1, 1]  # ns, we, nw, ne
    parameters = [(1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1), (1, 1, 2), \
                  (-1, -1, 2), (1, -1, 3), (-1, 1, 3)]
    for p in parameters:
        in_boundaries(grid, x, y, p, directions, match)
    is_winning = (max(directions) >= goal)
    if is_winning:
        moves = match[np.argmax(directions)]
        moves.append((x, y))
        value = grid[x, y] * 10
    return value, moves

good = 0
bad = 0

with open('train-1.renju', "r") as f:
    file_obj = open("bad_ones", "w")
    log = f.readline()
    while log:
        next = f.readline()
        while not next.startswith('black') and not next.startswith('white') and not next.startswith('draw') \
            and not next.startswith('unknown') and next:
            log = log[: -1]
            log += next
            next = f.readline()
        if log.startswith('draw') or log.startswith('unknown'):
            bad += 1
            log = next
            continue
        file_obj.write(log)
        good += 1
        log = next
    file_obj.close()
f.close()
print('okay: ', good / (good + bad) * 100)

