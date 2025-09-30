import random
import numpy as np
from numba import njit, types, typed
from tkinter import *
from time import *

class Board:
    def __init__(self, grid, turn=1):
        self.board = grid
        self.turn = turn
        self.legal_moves = []
        self.turn_over = dict()
        self.update_legal_moves()
        self.is_terminated = False

    def __repr__(self):
        return '\n'.join(str(i) for i in self.board)

    def isinrange(self, coordinates):
        return 0 <= coordinates[0] < 8 and 0 <= coordinates[1] < 8

    def update_legal_moves(self):
        validmoves = {(i, j): [] for i in range(8) for j in range(8)}
        validmoves2 = list(get_legal_moves_update(self.board, self.turn))
        for i in range(len(validmoves2) // 5):
            a, b, c, d, e = validmoves2[i * 5], validmoves2[i * 5 + 1], validmoves2[i * 5 + 2], validmoves2[
                i * 5 + 3], validmoves2[i * 5 + 4]
            validmoves[a, b].append((c, d, e))

        self.turn_over = validmoves
        self.legal_moves = [i for i in validmoves if len(validmoves[i]) > 0]

    def play_move(self, clx, cly, update=1):
        if (clx == -1 and cly == -1) or len(self.legal_moves) == 0:
            if len(self.legal_moves) == 0:
                self.turn = 3 - self.turn
                self.update_legal_moves()
                if len(self.legal_moves) == 0:
                    self.is_terminated = 1
            return -1

        if not len(self.turn_over.get((clx, cly), [])):
            return -1

        u = self.turn_over[clx, cly]
        cords = set()
        for row in u:
            dx, dy, length = row
            curx, cury = clx, cly
            cords.add(((curx, cury)))

            for i in range(length):
                curx += dx
                cury += dy
                cords.add((curx, cury))
        for i in cords:
            self.board[i[0]][i[1]] = self.turn
        self.turn = 3 - self.turn
        if update:
            self.update_legal_moves()
            if len(self.legal_moves) == 0:
                self.turn = 3 - self.turn
                self.update_legal_moves()
                if len(self.legal_moves) == 0:
                    self.is_terminated = 1

    def count_white_black(self):
        s1 = np.count_nonzero(self.board == 1)
        s2 = np.count_nonzero(self.board == 2)
        s0 = 64 - s1 - s2
        return (s0, s1, s2)

@njit
def isinrange(coordinates):
    return 0 <= coordinates[0] < 8 and 0 <= coordinates[1] < 8

@njit
def get_all_legal(grid, turn, i, j):
    L = [(x, y) for x in range(-1, 2) for y in range(-1, 2) if (x, y) != (0, 0)]
    answer = []
    for (dx, dy) in L:
        curx, cury = i, j
        curx += dx
        cury += dy
        fail = 0
        if not isinrange((curx, cury)) or grid[curx][cury] != 3 - turn:
            fail = 1
            continue
        for d in range(8):
            curx += dx
            cury += dy
            if not isinrange((curx, cury)) or grid[curx][cury] == turn:
                fail = 1
                break
            if grid[curx][cury] == 0:
                break
        if fail == 1:
            continue
        answer.append((curx, cury, -dx, -dy, d + 2))
    return answer

@njit
def get_legal_moves_update(grid, turn):
    validmoves = typed.List()
    for i in range(8):
        for j in range(8):
            if grid[i][j] != turn:
                continue
            answer = get_all_legal(grid, turn, i, j)
            for a, b, c, d, e in answer:
                validmoves.append(a)
                validmoves.append(b)
                validmoves.append(c)
                validmoves.append(d)
                validmoves.append(e)
    return list(validmoves)
INF = 10 ** 8
valgrid = np.array([
    [120, -20, 20, 5, 5, 20, -20, 120],[-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],[5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],[20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],[120, -20, 20, 5, 5, 20, -20, 120]
])
SEARCH_DEPTH = 0

@njit
def evaluate(grid, turn):
    s1, s2 = 0, 0
    empty_count = np.count_nonzero(grid == 0)
    is_endgame = empty_count <= 12

    for i in range(8):
        for j in range(8):
            if grid[i][j] == 1: s1 += valgrid[i][j]
            elif grid[i][j] == 2: s2 += valgrid[i][j]

    if is_endgame:
        c1 = np.count_nonzero(grid == 1)
        c2 = np.count_nonzero(grid == 2)
        #in endgame we only really care about the piece count, it's much more important than the strategic locations of pieces
        return (c1 - c2) * 10 if turn == 1 else (c2 - c1) * 10

    score = (s1 - s2) if turn == 1 else (s2 - s1)
    return score

def get_absolute_eval(grid):
    """Calculates evaluation from White's perspective (positive is good for White)."""
    s1, s2 = 0, 0
    for i in range(8):
        for j in range(8):
            if grid[i][j] == 1: s1 += valgrid[i][j]
            elif grid[i][j] == 2: s2 += valgrid[i][j]
    return s2 - s1

def negamax(board, alpha=-INF, beta=INF, depth=2):
    if depth == 0 or board.is_terminated:
        return (evaluate(board.board, board.turn), '', [])

    values, children_info = {}, []
    if len(board.legal_moves) == 0: return (-INF, (-1, -1), [])

    answer = -INF
    for move in board.legal_moves:
        board2 = Board(np.copy(board.board), turn=board.turn)
        board2.play_move(move[0], move[1], update=(depth > 1))
        combined, _, _ = negamax(board2, alpha=-beta, beta=-alpha, depth=depth - 1)
        current_score = -combined
        values[move] = current_score
        if depth == SEARCH_DEPTH:
            children_info.append({'move': move, 'score': current_score})
        answer = max(answer, current_score)
        alpha = max(alpha, answer)
        if (alpha >= beta): break

    if not values: return (-INF, (-1, -1), [])
    best_move = max(values, key=values.get)
    if depth == SEARCH_DEPTH:
        children_info.sort(key=lambda x: x['score'], reverse=True)
    return (values[best_move], best_move, children_info)

def AI_agent(board, depth=5):
    global SEARCH_DEPTH
    SEARCH_DEPTH = depth
    board_copy = Board(np.copy(board.board), board.turn)
    _, ans, tree_data = negamax(board_copy, depth=depth)
    return (ans, tree_data) if ans != -1 and ans is not None else ((-1, -1), [])

tk = Tk()
tk.title("Othello")
tk.state('zoomed') # Make window fullscreen

WIDTH = tk.winfo_screenwidth()
HEIGHT = tk.winfo_screenheight()
canvas = Canvas(tk, bg="#f0f0f0", height=HEIGHT, width=WIDTH, highlightthickness=0)
canvas.pack()
SIZE = min(WIDTH, HEIGHT) * 0.09
BOARD_OFFSET_X = (WIDTH - 10 * SIZE) / 2
BOARD_OFFSET_Y = (HEIGHT - 10 * SIZE) / 2
countsize = SIZE * 0.8
ringsize = SIZE * 0.7
ringsize2 = SIZE * 0.6
darkcolor = '#008060'
lightcolor = '#00a080'
ringcolor='#006666'
game_state = 'start_screen'
game_mode, player_black, player_white = None, None, None
board = None
lastx, lasty, clicked = 0, 0, 0
current_eval = 0

def draw_start_screen():
    canvas.delete("all")
    canvas.configure(bg="#005040")
    canvas.create_text(WIDTH / 2, HEIGHT * 0.2, font=('Segoe UI', 60, 'bold'), text="Othello", fill="white")
    canvas.create_rectangle(WIDTH / 2 - 200, HEIGHT * 0.4, WIDTH / 2 + 200, HEIGHT * 0.4 + 100, fill="#008060", tags="1p_button", outline="")
    canvas.create_text(WIDTH / 2, HEIGHT * 0.4 + 50, font=('Segoe UI', 28, 'bold'), text="Player vs AI", tags="1p_button", fill="white")
    canvas.create_rectangle(WIDTH / 2 - 200, HEIGHT * 0.6, WIDTH / 2 + 200, HEIGHT * 0.6 + 100, fill="#008060", tags="2p_button", outline="")
    canvas.create_text(WIDTH / 2, HEIGHT * 0.6 + 50, font=('Segoe UI', 28, 'bold'), text="Player vs Player", tags="2p_button", fill="white")

def start_game():
    global board, current_eval
    grid = np.array([[0] * 8 for _ in range(8)])
    grid[3][3], grid[4][4] = 2, 2
    grid[3][4], grid[4][3] = 1, 1
    board = Board(grid, 1)
    current_eval = get_absolute_eval(board.board)
    canvas.delete("all")
    canvas.configure(bg="#f0f0f0")
    full_redraw()

def full_redraw():
    draw_grid(BOARD_OFFSET_X, BOARD_OFFSET_Y, board.board, board.turn_over, SIZE)
    draw_game_info()
    draw_eval_bar(current_eval)
    if game_mode == '1P':
        is_ai_turn = (board.turn == 1 and player_black != 'Human') or (board.turn == 2 and player_white != 'Human')
        if not is_ai_turn:
             draw_tree([])
def draw_grid(posx, posy, grid, validmoves, ssize):
    canvas.delete("grid")
    for i in range(1, 9):
        for j in range(1, 9):
            xc, yc = posx + i * ssize, posy + j * ssize
            f8 = (darkcolor if (i + j) % 2 == 1 else lightcolor)
            canvas.create_rectangle(xc, yc, xc + ssize, yc + ssize, fill=f8, tags="grid", outline="#004030")
            u = grid[i - 1][j - 1]
            if u in [1, 2]:
                color = 'black' if u == 1 else 'white'
                canvas.create_oval(xc + ssize / 2 - countsize / 2, yc + ssize / 2 - countsize / 2,
                                   xc + ssize / 2 + countsize / 2, yc + ssize / 2 + countsize / 2, fill=color,
                                   tags="grid", outline="")
            if validmoves.get((i - 1, j - 1), []):
                captures = sum(line[-1] - 1 for line in validmoves.get((i - 1, j - 1), []))
                canvas.create_oval(xc + ssize / 2 - ringsize / 2, yc + ssize / 2 - ringsize / 2,
                                   xc + ssize / 2 + ringsize / 2, yc + ssize / 2 + ringsize / 2,
                                   fill=lightcolor, tags="grid", outline="")
                canvas.create_text(xc + ssize / 2, yc + ssize / 2, font=('Segoe UI', int(SIZE * 0.3), 'bold'),
                                   text=str(captures), fill="white", tags="grid")
def draw_tree(tree_data):
    canvas.delete("tree_text")
    tree_x = BOARD_OFFSET_X + 10 * SIZE
    canvas.create_rectangle(tree_x, BOARD_OFFSET_Y, WIDTH - 40, BOARD_OFFSET_Y + 8 * SIZE + 2*SIZE, fill="#e0e0e0", outline="", tags="tree_text")
    canvas.create_text(tree_x + 20, BOARD_OFFSET_Y + 40, font=('Segoe UI', 16, 'bold'), text="AI Move Evaluation", tags="tree_text", anchor='w')
    if not tree_data: return
    start_y = BOARD_OFFSET_Y + 80
    for i, node in enumerate(tree_data[:20]):
        move, score = node['move'], node['score']
        text = f"Move {move}:  Score {score:.2f}"
        color = "#c00000" if i == 0 else "black"
        canvas.create_text(tree_x + 20, start_y + i * 30, font=('Segoe UI', 12), text=text, anchor='w', fill=color, tags="tree_text")

def draw_eval_bar(evaluation):
    canvas.delete("eval_bar")
    bar_x, bar_y = 40, BOARD_OFFSET_Y + SIZE
    bar_width, bar_height = 40, 8 * SIZE
    max_eval = 300 
    canvas.create_rectangle(bar_x, bar_y, bar_x + bar_width, bar_y + bar_height, fill="#101010", outline="", tags="eval_bar")
    normalized_score = (evaluation + max_eval) / (2 * max_eval)
    white_height = max(0, min(bar_height, normalized_score * bar_height))
    canvas.create_rectangle(bar_x, bar_y + bar_height - white_height, bar_x + bar_width, bar_y + bar_height, fill="white", outline="", tags="eval_bar")
def draw_game_info():
    canvas.delete("info_text")
    info_y = BOARD_OFFSET_Y - 40
    canvas.create_text(BOARD_OFFSET_X + 4.5 * SIZE, info_y - 20, font=('Segoe UI', 14), text=f"Black: {player_black} vs White: {player_white}", tags="info_text", anchor='center')
    _, b_score, w_score = board.count_white_black()
    score_str = f"Black: {b_score}  |  White: {w_score}"
    canvas.create_text(BOARD_OFFSET_X + 4.5 * SIZE, BOARD_OFFSET_Y + 9.5 * SIZE, font=('Segoe UI', 24, 'bold'), text=score_str, tags="info_text", anchor='center')

    if board.is_terminated:
        winner_text = "Draw!" if b_score == w_score else "Black Wins!" if b_score > w_score else "White Wins!"
        margin = abs(b_score - w_score)
        final_text = f"Game Over: {winner_text} by {margin} pieces"
        canvas.create_text(BOARD_OFFSET_X + 4.5 * SIZE, info_y + 20, font=('Segoe UI', 28, 'bold'), text=final_text, fill="#00529B", tags="info_text", anchor='center')
    else:
        turn_text = "Black's Turn" if board.turn == 1 else "White's Turn"
        turn_color = "black" if board.turn == 1 else "gray20"
        canvas.create_text(BOARD_OFFSET_X + 4.5 * SIZE, info_y + 20, font=('Segoe UI', 28, 'bold'), text=turn_text, fill=turn_color, tags="info_text", anchor='center')
def get_click_on_grid(clickxc, clickyc):
    return (int((clickxc - BOARD_OFFSET_X) // SIZE - 1), int((clickyc - BOARD_OFFSET_Y) // SIZE - 1))
def handle_click(event):
    global lastx, lasty, clicked, game_state, game_mode, player_black, player_white
    lastx, lasty = event.x, event.y

    if game_state == 'start_screen':
        if WIDTH / 2 - 200 < lastx < WIDTH / 2 + 200:
            if HEIGHT * 0.4 < lasty < HEIGHT * 0.4 + 100:
                game_mode, player_black, player_white = '1P', 'Human', 'Negamax AI'
                game_state = 'in_game'
                start_game()
            elif HEIGHT * 0.6 < lasty < HEIGHT * 0.6 + 100:
                game_mode, player_black, player_white = '2P', 'Human', 'Human'
                game_state = 'in_game'
                start_game()
    elif game_state == 'in_game':
        clicked = 1
canvas.bind("<Button-1>", handle_click)
while True:
    if game_state == 'start_screen':
        draw_start_screen()
    elif game_state == 'in_game' and not board.is_terminated:
        is_human_turn = (board.turn == 1 and player_black == 'Human') or (board.turn == 2 and player_white == 'Human')
        
        if is_human_turn and clicked:
            clicked = 0
            clx, cly = get_click_on_grid(lastx, lasty)
            if (clx, cly) in board.legal_moves:
                board.play_move(clx, cly)
                current_eval = get_absolute_eval(board.board)
                full_redraw()
                tk.update()

        elif (board.turn == 2 and player_white != 'Human') or (board.turn == 1 and player_black != 'Human'):
            (clx, cly), tree = AI_agent(board, depth=5)
            board.play_move(clx, cly)
            current_eval = get_absolute_eval(board.board)
            full_redraw()
            draw_tree(tree)
        
        if len(board.legal_moves) == 0 and not board.is_terminated:
            board.play_move(-1, -1)
            full_redraw()

    tk.update_idletasks()
    tk.update()
    sleep(0.01)
