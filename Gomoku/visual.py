import numpy as np
import random
from tkinter import *
import random
from constants import *
from formating import *
from cnn import *
global log
global moves
global player
global first
global second
step = 0


moves = []


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



def make_move(log):
    st = time.time()
    log = log.strip()
    if log == '':
        starters = ["g7", "h7", "j7", "g8", "h8", "j8", "g9", "h9", "j9"]
        return format_move(random.choice(starters))
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
    move_defence = (0, 0) #delete later
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
    return mv


def get_moves_upd(log):
    if log[-1] == '\n':
        log = log[:-1]
    log = log.split()
    for i in range(len(log)):
        log[i] = format_move(log[i])
    return log


clicked = np.zeros((n, n), dtype=np.int8)

policy = dummy

global prev_x, prev_y
prev_x, prev_y = -1, -1

def click(event):
    global prev_x, prev_y
    global moves
    global player
    player = 3 - player
    ids = -1
    ids = c.find_withtag(CURRENT)[0]
    x = (ids - 1) % n
    y = (ids - 1) // n
    clicked[x][y] = 1
    gameover, _ = is_winning_move(x, y, clicked)
    moves.append((x + 1, y + 1))
    c.itemconfig(CURRENT, fill=first)
    log = get_log(moves, '')
    c.update()
    if gameover:
        print("Player " + str(player) + " wins")
        clear()
        print ("Do you want to play again? (y/n)")
        ans = input()
        if ans == "y":
            main()
        else:
            root.destroy()
    x, y = make_move(log)
    moves.append((x, y))
    x -= 1
    y -= 1
    ids = y * n + x + 1
    print("(x, y):", x, y)
    clicked[x][y] = -1
    c.itemconfig(ids, fill=second)
    c.update()
    gameover, _ = is_winning_move(x, y, clicked)
    player = 3 - player
    if gameover:
        print("Player " + str(player) + " wins")
        clear()
        print ("Do you want to play again? (y/n)")
        ans = input()
        if ans == "y":
            main()
        else:
            root.destroy()

def clear():
    for i in range(n):
        for j in range(n):
            c.create_rectangle(i * side, j * side,
                               i * side + side,
                               j * side + side, fill=white)
    c.update()

def main():
    global first
    global second
    global moves
    global log
    log = ''
    clear()
    print("black or white? (b/w)")
    ans = input()
    global player
    player = 2
    moves = []
    if ans == 'w':
        player = 3 - player
        first, second = marigold,
        x, y = make_move(log)
        moves.append((x, y))
        x -= 1
        y -= 1
        ids = y * n + x + 1
        print("(x, y):", x, y)
        clicked[x][y] = -1
        c.itemconfig(ids, fill=second)
        c.update()
    root.mainloop()

if __name__ == "__main__":
    root = Tk()
    root.title("連珠")
    c = Canvas(root, width=n * side,
               height=n * side)
    c.pack()

    c.bind("<Button-1>", click)
    global first
    global second
    first, second = ciric, marigold
    main()