import logging
import os
import random

import backend
import numpy
import renju

import numpy as np
import random
import time
import torch
from torch import utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

n = 15
goal = 5



def get_log(moves, winner):
    log = winner
    for move in moves:
        log += ' '
        log += format_from_move(move)
    log += '\n'
    return log

let_to_dig = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, \
              'h': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15}

dig_to_let = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', \
              8: 'h', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p'}

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.norm1 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.norm3 = torch.nn.BatchNorm2d(96)
        self.conv6 = torch.nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.norm4 = torch.nn.BatchNorm1d(128 * 4 * 4)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 30 * 30)
        self.norm5 = torch.nn.BatchNorm1d(30 * 30)
        self.fc2 = torch.nn.Linear(30 * 30, 15 * 15)

    def forward(self, x):
        x = x.view(-1, 3, 16, 16)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.norm1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.norm2(x)
        x = F.relu(self.conv5(x))
        x = self.norm3(x)
        x = F.relu(self.conv6(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.norm4(x)
        x = F.relu(self.fc1(x))
        x = self.norm5(x)
        x = self.fc2(x)
        return (x)


model = CNN()
model.load_state_dict(torch.load("upd2", map_location='cpu'))
model.eval()

def format_move(old_move):
    y, x = let_to_dig[old_move[0]], int(old_move[1:])
    new_move = (x, y)
    return new_move


def format_from_move(move):
    x, y = move
    res = dig_to_let[y]
    res += str(x)
    return res


def get_moves_upd(log):
    if log == '':
        return []
    if log[-1] == '\n':
        log = log[:-1]
    log = log.split()
    for i in range(len(log)):
        log[i] = format_move(log[i])
    return log


def get_grid(log):
    moves = get_moves_upd(log)
    #print(moves)
    grid_black = torch.zeros((n + 1, n + 1), dtype=torch.int8)
    grid_white = torch.zeros((n + 1, n + 1), dtype=torch.int8)
    grid_player = torch.ones((n + 1, n + 1), dtype=torch.int8)
    grid_player *= -1
    mvs_num = len(moves)
    #print("mvs_num:", mvs_num)
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
    return (lbl - 1) // n + 1, (lbl - 1) % n + 1


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


def open_three(x, y, grid):
    match = [[], [], [], []]
    ends = [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]
    directions = [1, 1, 1, 1]  # ns, we, nw, ne
    parameters = [(1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1), (1, 1, 2), \
                  (-1, -1, 2), (1, -1, 3), (-1, 1, 3)]
    for p in parameters:
        in_boundaries(grid, x, y, p, directions, match)
    streak = max(directions)
    for d in range(len(directions)):
        if directions[d] >= goal - 2:
            moves = match[d]
            moves.append((x, y))
            moves.sort()
            open = 0
            x_b, y_b = moves[-1][0] + parameters[2 * d][0], moves[-1][1] + parameters[2 * d][1]
            x_e, y_e = moves[0][0] + parameters[2 * d + 1][0], moves[0][1] + parameters[2 * d + 1][1]
            if 0 < x_b <= n and 0 < y_b <= n and not grid[x_b, y_b]:
                open += 1
                ends[d][0] = (x_b, y_b)
            if 0 < x_e <= n and 0 < y_e <= n and not grid[x_e, y_e]:
                open += 1
                ends[d][1] = (x_e, y_e)
    return ends, streak, np.argmax(directions)



def make_move(log, grid=None):
    st = time.time()
    log = log.strip()
    if log == '':
        starters = ["g7", "h7", "j7", "g8", "h8", "j8", "g9", "h9", "j9"]
        return random.choice(starters)
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
    ends, streak_defence, _ = open_three(last_x, last_y, flat_grid)
    if ends != [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]:
        positions = [p for pos in ends for p in pos if p != (-1, -1)]
        if move_net in positions:
            move_defence = move_net
        else:
            move_defence = random.choice(positions)
        mv = move_defence
    move_attack = move_net
    streak_attack = 0
    for move in plr_moves:
        if time.time() - st > 2.6:
            break
        ends, str_attack, d = open_three(move[0], move[1], flat_grid)
        streak_attack = max(streak_attack, str_attack)
        if ends != [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]:
            positions = [p for pos in ends for p in pos if p != (-1, -1)]
            if move_net in positions:
                break
            move_attack = random.choice(positions)
    if streak_attack >= streak_defence:
        mv = move_attack
    if streak_attack < 3 and streak_defence < 3:
        mv = move_net
    return format_from_move(mv)


def choose_random_move(board):
    positions = renju.list_positions(board, renju.Player.NONE)
    return renju.to_move(random.choice(positions))

def main():
    pid = os.getpid()
    LOG_FORMAT = str(pid) + ':%(levelname)s:%(asctime)s: %(message)s'

    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    logging.debug("Start dummy backend...")
    try:
        while True:
            logging.debug("Wait for game update...")
            game = backend.wait_for_game_update()

            if not game:
                logging.debug("Game is over!")
                return
            logging.debug('Game: [%s]', game.dumps())
            move = make_move(game.dumps())
            if not backend.set_move(move):
                logging.error("Impossible set move!")
                return

            logging.debug('Random move: %s', move)

    except:
        logging.error('Error!', exc_info=True, stack_info=True)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = CNN()
    model.load_state_dict(torch.load("upd3", map_location='cpu'))
    model.eval()
    main()