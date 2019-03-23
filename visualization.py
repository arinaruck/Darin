from constants import *
from n_depth_tree import *
from formating import *
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Renju")
font = pygame.font.Font(font_name, 36, bold=True)


def text_objects(msg, font, col=black):
    textSurface = font.render(msg, True, col)
    return textSurface, textSurface.get_rect()


def message_display(msg, pos, col=black, font_size=65):
    font = pygame.font.Font(font_name, font_size, bold=True)
    TextSurf, TextRect = text_objects(msg, font, col)
    TextRect.center = pos
    screen.blit(TextSurf, TextRect)
    pygame.display.update()


def button(msg, x, y, w, h, inact_col, act_col, text_col=white, \
           font_size=35, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if intersects(mouse, x, y, x + w, y + h):
        pygame.draw.rect(screen, act_col, (x, y, w, h))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(screen, inact_col, (x, y, w, h))
    message_display(msg, (x + w // 2, y + h // 2), col=text_col, font_size=font_size)


def draw_field(grid):
    for row in range(n):
        for column in range(n):
            color = colors[grid[row][column]]
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + width) * column + MARGIN,
                              (MARGIN + height) * row + MARGIN,
                              width, height])

    pygame.display.flip()


def quit_game():
    pygame.quit()
    quit()


def game_intro():
    clock = pygame.time.Clock()
    intro = True
    while intro:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game()
        screen.fill(white)
        message_display("Rendju", (window_side // 2, window_side // 2 - 50), col=font_col)
        message_display("Game", (window_side // 2, window_side // 2 + 10), col=font_col)
        mouse = pygame.mouse.get_pos()
        button("Play", x_b_1, y_b_1, b_w, b_h, button_1_col, button_1_col_pushed, action=game_loop)
        button("Quit", x_b_2, y_b_2, b_w, b_h, button_2_col, button_2_col_pushed, action=quit_game)
        pygame.display.update()
        clock.tick(25)


def game_loop():
    grid = np.zeros((n, n), dtype=np.int8)
    game_exit = False
    gameover = False
    clock = pygame.time.Clock()
    move_num = 0
    player = 1
    screen.fill(black)
    pygame.display.update()
    while not game_exit:
        if gameover:
            game_over(player)
            print("Player ", player, "won the game")
            gameover = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                column = pos[0] // (width + MARGIN)
                row = pos[1] // (height + MARGIN)
                if grid[row][column]:
                    continue
                player *= -1
                print(format_from_move((row, column)), end=' ')
                grid[row][column] = player
                value, moves = is_winning_move(row, column, grid)
                if value != 0:
                    gameover = True
                    print(gameover, moves)
                    print("Player ", player, " wins")
                    colors[3] = ((min(255, colors[player][0] // 1.25), min(255, colors[player][1] // 1.25), \
                                  min(255, colors[player][2] // 1.25)))
                    print(len(colors), colors[3])
                    for move in moves:
                        grid[move] = 3
                        draw_field(grid)
                move_num += 1
                #print("Click ", pos, "Grid coordinates: ", row, column)
            '''else:
                move, game_exit = find_best_move(grid, player)
                grid[move] = player
                print(move, player)
                player = -player
            '''
        screen.fill(black)
        if not game_exit:
            draw_field(grid)
        if gameover:
            print("it's over, bro")
            game_over(player, grid)
            break
        pygame.display.update()
        clock.tick(60)


def game_over(player, grid):
    clock = pygame.time.Clock()
    over = True
    msg = "Player " + str(player) + " wins"
    print(grid)
    while over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        draw_field(grid)
        message_display(msg, (window_side // 2, window_side // 2), font_col, 55)
        mouse = pygame.mouse.get_pos()
        button("Play", x_b_1, y_b_1, b_w, b_h, button_1_col, button_1_col_pushed, action=game_loop)
        button("Quit", x_b_2, y_b_2, b_w, b_h, button_2_col, button_2_col_pushed, action=quit_game)
        pygame.display.update()
        clock.tick(25)

game_intro()