# import dependencies
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

import neat
import pickle

import pygame
import time


# initalize constants
FPS = 60
HEIGHT = 500
WIDTH = 500

BLOCK_SIZE = 20
SIDE_BUFFER = 135
TOP_BUFFER = 75
BUFFER = 40

all_blocks = [
    [[0, 0, 0, 0], [1, 1, 1, 1], 0],
    [[1, 0, 0, 0], [1, 1, 1, 0], 1],
    [[0, 0, 0, 1], [0, 1, 1, 1], 2],
    [[0, 1, 1, 0], [0, 1, 1, 0], 3],
    [[0, 0, 1, 1], [0, 1, 1, 0], 4],
    [[0, 0, 1, 0], [0, 1, 1, 1], 5],
    [[1, 1, 0, 0], [0, 1, 1, 0], 6]
]

colour_dict = {0: (0, 204, 204),
               1: (0, 0, 204),
               2: (255, 128, 0),
               3: (255, 255, 0),
               4: (0, 255, 0),
               5: (204, 0, 204),
               6: (255, 0, 0)
              }

GRAY = (50, 50, 50)
DARK_GRAY = (32, 32, 32)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WHITE = (255, 255, 255)

runs_per_net = 5
high_score = 0


class TetrisEnv(gym.Env):
    def __init__(self, display=False):
        pygame.init()

        # if display is true render the GUI
        self.display = display
        if self.display:
            self.WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Tetris")
            self.clock = pygame.time.Clock()

        # returns 4 weights which are used to assess the current board state
        # these 4 weights are: aggregate height, # lines completed, # holes, bumpiness
        self.action_space = Box(0, 1, shape=(4,))

        # a 10x20 tetris board has too many different possibilities so we pass in a simplified board instead
        # observation space (1d vector) --> curr_block, held block, next 3 blocks, height of each col, # holes in each col
        self.observation_space = Box(0, 20, shape=(25,), dtype=int)

        self.iterations = 0

        self.reset()

    # flattens everything into a 1d vector which is used as the current state of the board
    def convert_to_state(self, block_type, hold, next_blocks, height, holes):
        state = [block_type[-1], hold[-1]]

        for block in next_blocks:
            state.append(block[-1])

        state += height + holes

        return state

    # returns the height of each col, # of completed lines, holes in each col, and bumpiness
    def get_heuristics(self, board):
        # initialize height and holes
        height = [0] * len(board[0])
        holes = [0] * len(board[0])

        # find height and # of holes for each col
        for col in range(len(board[0])):
            for row in range(len(board)):
                if board[row][col] == 1 and height[col] == 0:
                    height[col] = len(board) - row
                if height[col] != 0 and board[row][col] == 0:
                    holes[col] += 1

        # calculate bumpiness
        # this is done by taking the difference in height between all adjacent columns
        bumpiness = 0
        for col in range(len(height) - 1):
            bumpiness += abs(height[col] - height[col + 1])

        # calculate number of completed rows
        rows_cleared = 0
        for row in range(len(board)):
            if 0 not in board[row]:
                rows_cleared += 1

        return height, rows_cleared, holes, bumpiness

    # rotates the block
    def rotate_block(self, block, board):
        # create the new rotated block
        rotated_block = []
        for col in range(len(block[0])):
            new_row = []
            for row in range(len(block)):
                new_row.append(block[row][col])
            rotated_block.append(new_row)

        # removes any unecessary white space e.g. empty cols and empty rows
        rotated_block = self.reshape_block(rotated_block)
        return rotated_block

    # hard drops block
    def hard_drop(self, block, x, y, board, board_colours):
        # lowers the blocks y coord until it should be placed --> hits the bottom of the board or another block
        while self.valid(block, x, y + 1, board):
            y += 1
        # updates board
        new_board, new_board_colours = self.update_board(block, x, y, self.make_copy(board),
                                                         self.make_copy(board_colours))
        return new_board, new_board_colours

    # helper function
    # used to maky a copy of blocks, board, or other 2d arrays
    def make_copy(self, item):
        copied = []
        for row in item:
            copied.append(row.copy())
        return copied

    # analyzes all possible places the current block can be placed and assess that board state using the given weights
    # then returns the best move/board state out of the given options
    def get_best_move(self, block_type, hold, next_blocks, board, board_colours, action):
        # make a copy of the block, next block, held block, and board
        block = self.make_copy(self.reshape_block(block_type[:len(block_type) - 1]))
        next_block = self.make_copy(self.reshape_block(next_blocks[0][:len(next_blocks[0]) - 1]))
        hold_block = self.make_copy(self.reshape_block(hold[:len(hold) - 1]))
        if not hold_block:
            hold_block = self.make_copy(self.reshape_block(next_blocks[0][:len(next_blocks[0]) - 1]))

        new_board = self.make_copy(board)

        # get the weights of each heuristic
        # since we want our tower to be short, minimize holes, and minimize bumpiness we can assume that these weights are
        # negative
        # since we want to clear as many lines as possible we can assume that line weight is positive
        height_weight = -1 * action[0]
        line_weight = action[1]
        hole_weight = -1 * action[2]
        bump_weight = -1 * action[3]

        # initialize best move, best score, and whether or not we should hold the block
        best_move = None
        best_sore = None
        hold = False

        # find all possible places we can place the current block meaning each rotation and each column that we can place the
        # block in
        # loop through all valid rotations
        for rotation in range(4):
            # rotate the block
            if rotation != 0:
                block = self.rotate_block(block, new_board)
            # loop through all valid columns
            for col in range(len(board[0])):
                if col + len(block[0]) > len(board[0]):
                    break
                # update curr board and get the heuristics for this board
                curr_board, new_board_colours = self.hard_drop(block, col, 0, new_board, self.make_copy(board_colours))
                height, lines, holes, bumpiness = self.get_heuristics(curr_board)
                # calculate score using given weights
                score = height_weight * (
                            sum(height) - (lines * len(board[0]))) + line_weight * lines + hole_weight * sum(
                    holes) + bump_weight * bumpiness

                # if the current move is better that our existing best move update the best move
                if best_move is None or score > best_sore:
                    # best move contains 2 values the rotation the block should be and the column to place the block
                    best_move = [rotation, col]
                    best_sore = score

        # determine whether or not the current block should be held
        # to determine this we simply calculate the best score achievable by the best block and compare it to the best
        # score achievable by the current block, if the held block achieves a higher score we switch the current and held block
        for rotation in range(4):
            if rotation != 0:
                hold_block = self.rotate_block(hold_block, new_board)
            for col in range(len(board[0])):
                if col + len(hold_block[0]) > len(board[0]):
                    break
                # update board assuming the current block is the held block
                curr_board, new_board_colours = self.hard_drop(hold_block, col, 0, new_board,
                                                               self.make_copy(board_colours))
                height, lines, holes, bumpiness = self.get_heuristics(curr_board)
                # calculate score
                score = height_weight * (
                            sum(height) - (lines * len(board[0]))) + line_weight * lines + hole_weight * sum(
                    holes) + bump_weight * bumpiness
                # if the score achieved by the move is higher than the current move we set hold to True and update the best move
                if best_move is None or score > best_sore:
                    best_move = [rotation, col]
                    best_sore = score
                    hold = True

        return best_move, hold

    # delete empty rows and col in a block e.g. trims it down
    # helpful as each block is represented by a 2x4 grid but certain blocks will use up more space than others
    def reshape_block(self, block):
        # make copy of original block before reshaping it
        new_block = self.make_copy(block)
        col_delete = []
        row_delete = []

        # delete blank cols
        count = 0
        for col in range(len(new_block[0])):
            delete = True
            for row in range(len(new_block)):
                if new_block[row][col - count] == 1:
                    delete = False
                    break
            if delete:
                for row in range(len(new_block)):
                    new_block[row].pop(col - count)
                count += 1

        # delete blank rows
        count = 0
        for row in range(len(new_block)):
            if new_block[row - count] == [0] * len(new_block[row - count]):
                new_block.pop(row - count)
                count += 1

        return new_block

    # draw current_block onto the board once it's placed
    def update_board(self, block, x, y, board, board_colours):
        for row in range(len(block)):
            for col in range(len(block[0])):
                if block[row][col] == 0:
                    continue
                board[y + row][x + col] = 1
                board_colours[y + row][x + col] = self.block_type[-1]

        return board, board_colours

    # check if a block's position is valid
    def valid(self, block, x, y, board):
        # check if block is within bounds of the grid
        if 0 <= x < x + len(block[0]) <= len(self.board[0]) and 0 <= y < y + len(block) <= len(self.board):
            # check if the block is overlapping a pre placed block
            for row in range(len(block)):
                for col in range(len(block[0])):
                    if block[row][col] == 1 and board[y + row][x + col] != 0:
                        return False
            return True
        return False

    # generate a new block from current block pouch
    def generate_new_block(self, block_pouch):
        next_block = random.choice(block_pouch)
        block_pouch.remove(next_block)
        return next_block

    # refill block pouch
    def fill_block_pouch(self):
        pouch = []
        for block in all_blocks:
            pouch.append(block)
        return pouch

    # update curr_block --> used when curr_block is replaced by the first block of next_blocks
    def update_curr_block(self, block):
        self.block_type = block.copy()
        self.curr_block = self.reshape_block(block[:len(block) - 1])
        self.rotation = 0
        self.curr_y = 0
        self.curr_x = (len(self.board[0]) - len(self.curr_block[0])) // 2

    # clears filled rows
    def clear_row(self, board, board_colours):
        rows_cleared = 0
        # if a row is full, pop that row and append it to the top of the board
        for row in range(len(self.board)):
            if 0 not in self.board[row]:
                board.pop(row)
                board_colours.pop(row)
                board.insert(0, [0] * 10)
                board_colours.insert(0, [-1] * 10)
                rows_cleared += 1

        return rows_cleared, board, board_colours

    # determine whether or not the player has lost
    def game_over(self, block, x, y, board):
        if y == 0 and not self.valid(block, x, y, board):
            return True
        return False

    def step(self, action):
        # render GUI,
        if self.display:
            self.render()

        reward = 0

        # applies the same set of weights for the next 4 moves as the current block, and the next three blocks are given
        for i in range(4):
            # determine the best move
            best_move, hold = self.get_best_move(self.block_type, self.hold, self.next_blocks, self.board,
                                                 self.board_colours, action)
            self.curr_block = self.make_copy(self.reshape_block(self.block_type[:len(self.block_type) - 1]))

            # replace curr_block and the held block if the held block results in a better board state
            if hold:
                if self.hold[-1] == -1:
                    self.hold = self.block_type
                    self.update_curr_block(self.next_blocks.pop(0))
                    self.next_blocks.append(self.generate_new_block(self.block_pouch))
                    if len(self.block_pouch) == 0:
                        self.block_pouch = self.fill_block_pouch()
                else:
                    self.block_type, self.hold = self.hold, self.block_type
                    self.update_curr_block(self.block_type)

            # rotate the block until it matches the best move's block orientation
            if best_move[0] != self.rotation:
                for rotation in range(4):
                    if rotation != 0:
                        self.curr_block = self.rotate_block(self.curr_block, self.board)
                    if rotation == best_move[0]:
                        break
                    # shift the block down by 1 space if possible
                    if self.valid(self.curr_block, self.curr_x, self.curr_y + 1, self.board):
                        self.curr_y += 1

                    # render GUI
                    if self.display:
                        self.render()

            # shift the block left until it matches the best move's column
            if best_move[1] < self.curr_x:
                while best_move[1] < self.curr_x:
                    if self.valid(self.curr_block, self.curr_x - 1, self.curr_y, self.board):
                        self.curr_x -= 1
                    else:
                        break

                    # shift the block down by 1 space if possible
                    if self.valid(self.curr_block, self.curr_x, self.curr_y + 1, self.board):
                        self.curr_y += 1
                    # render GUI
                    if self.display:
                        self.render()

            # shift the block right until it matches the best move's column
            elif best_move[1] > self.curr_x:
                while best_move[1] > self.curr_x:
                    if self.valid(self.curr_block, self.curr_x + 1, self.curr_y, self.board):
                        self.curr_x += 1
                    else:
                        break

                    # shift the block down by 1 space if possible
                    if self.valid(self.curr_block, self.curr_x, self.curr_y + 1, self.board):
                        self.curr_y += 1

                    # render GUI
                    if self.display:
                        self.render()

            # hard drop
            # continously drop block until it can no longer be dropped
            self.board, self.board_colours = self.hard_drop(self.curr_block, self.curr_x, self.curr_y,
                                                            self.board, self.board_colours)

            # update curr_block, next_blocks, and block pouch
            self.update_curr_block(self.next_blocks.pop(0))
            self.next_blocks.append(self.generate_new_block(self.block_pouch))
            if len(self.block_pouch) == 0:
                self.block_pouch = self.fill_block_pouch()

            # update reward and score
            # reward is the same as the tetris scoring function --> +10 for block placed, +40 for 1 line cleared,
            # + 100 for 2 lines cleared, + 300 for 3 lines cleared, + 1200 for 4 lines cleared
            reward += 10
            self.score += 10

            # clear any filled rows
            rows_cleared, self.board, self.board_colours = self.clear_row(self.board, self.board_colours)

            # update reward based on the number of rows cleared
            if rows_cleared > 0:
                self.rows_cleared += rows_cleared
                if rows_cleared == 1:
                    self.score += 40
                    reward += 40
                elif rows_cleared == 2:
                    self.score += 100
                    reward += 100
                elif rows_cleared == 3:
                    self.score += 300
                    reward += 300
                else:
                    self.score += 1200
                    reward += 1200

            # check if game is finished
            done = self.game_over(self.curr_block, self.curr_x, self.curr_y, self.board)
            # if game over give a slight punishment to the agent
            if done:
                reward -= 5
                break

        # update state
        info = {}
        self.state = self.convert_to_state(self.block_type, self.hold, self.next_blocks, self.height, self.holes)

        return np.array(self.state), reward, done, info

    def render(self):
        self.WINDOW.fill(GRAY)
        for event in pygame.event.get():
            self.clock.tick(FPS)
            # close window when exit button is clicked
            if event.type == pygame.QUIT:
                pygame.quit()

        # draw board
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                # determine colours
                # empty
                if self.board[row][col] == 0:
                    colour = DARK_GRAY
                # not empty
                else:
                    colour = colour_dict[self.board_colours[row][col]]
                # draw blocks
                pygame.draw.rect(self.WINDOW, colour, (SIDE_BUFFER + BLOCK_SIZE * col,
                                                       TOP_BUFFER + BLOCK_SIZE * row, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.WINDOW, BLACK, (SIDE_BUFFER + BLOCK_SIZE * col,
                                                      TOP_BUFFER + BLOCK_SIZE * row, BLOCK_SIZE, BLOCK_SIZE), 1)

        # draw curr_block
        for row in range(len(self.curr_block)):
            for col in range(len(self.curr_block[0])):
                if self.curr_block[row][col] == 0:
                    continue
                pygame.draw.rect(self.WINDOW, colour_dict[self.block_type[-1]],
                                 (SIDE_BUFFER + BLOCK_SIZE * (col + self.curr_x),
                                  TOP_BUFFER + BLOCK_SIZE * (row + self.curr_y), BLOCK_SIZE, BLOCK_SIZE))

                pygame.draw.rect(self.WINDOW, BLACK, (SIDE_BUFFER + BLOCK_SIZE * (col + self.curr_x),
                                                      TOP_BUFFER + BLOCK_SIZE * (row + self.curr_y), BLOCK_SIZE,
                                                      BLOCK_SIZE),
                                 1)

        # draw hold
        # display hold text
        FONT = pygame.font.Font('freesansbold.ttf', 20)
        text = FONT.render(f"Hold", True, WHITE)
        self.WINDOW.blit(text, (BUFFER, TOP_BUFFER))

        # display hold block
        reshaped_hold = self.reshape_block(self.hold[:len(self.hold) - 1])
        for row in range(len(reshaped_hold)):
            for col in range(len(reshaped_hold[0])):
                if reshaped_hold[row][col] == 0:
                    continue

                pygame.draw.rect(self.WINDOW, colour_dict[self.hold[-1]],
                                 (BUFFER + BLOCK_SIZE * col,
                                  TOP_BUFFER + BUFFER + BLOCK_SIZE * row, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.WINDOW, BLACK,
                                 (BUFFER + BLOCK_SIZE * col,
                                  TOP_BUFFER + BUFFER + BLOCK_SIZE * row, BLOCK_SIZE, BLOCK_SIZE), 1)

        # draw next_blocks
        # display next blocks text
        text = FONT.render(f"Next Blocks", True, WHITE)
        self.WINDOW.blit(text, (SIDE_BUFFER + BLOCK_SIZE * len(self.board[0]) + BUFFER / 2, TOP_BUFFER))

        # display next_blocks
        for idx, block in enumerate(self.next_blocks):
            new_block = self.reshape_block(block[:len(block) - 1])
            if len(new_block) == 1:
                new_block.append([0] * 4)
            for row in range(len(new_block)):
                for col in range(len(new_block[0])):
                    if new_block[row][col] == 0:
                        continue
                    pygame.draw.rect(self.WINDOW, colour_dict[block[-1]],
                                     (SIDE_BUFFER + BLOCK_SIZE * len(self.board[0]) + BUFFER / 2 + BLOCK_SIZE * col,
                                      TOP_BUFFER + BUFFER + BLOCK_SIZE * row + idx * (
                                              len(new_block) * BLOCK_SIZE + BUFFER),
                                      BLOCK_SIZE, BLOCK_SIZE))

                    pygame.draw.rect(self.WINDOW, BLACK,
                                     (SIDE_BUFFER + BLOCK_SIZE * len(self.board[0]) + BUFFER / 2 + BLOCK_SIZE * col,
                                      TOP_BUFFER + BUFFER + BLOCK_SIZE * row + idx * (
                                              len(new_block) * BLOCK_SIZE + BUFFER),
                                      BLOCK_SIZE, BLOCK_SIZE), 1)

        # display score and iterations
        FONT = pygame.font.Font('freesansbold.ttf', 16)
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.WINDOW.blit(text, (SIDE_BUFFER + BLOCK_SIZE * len(self.board[0]) + BUFFER / 2, 380))

        text = FONT.render(f"Iterations: {self.iterations}", True, WHITE)
        self.WINDOW.blit(text, (SIDE_BUFFER + BLOCK_SIZE * len(self.board[0]) + BUFFER / 2, 430))

        pygame.display.update()

    def reset(self):

        # initlaize an empty board and empty hold
        self.board = []
        self.board_colours = []
        for i in range(20):
            self.board.append([0] * 10)
            self.board_colours.append([-1] * 10)
        self.hold = [[0, 0, 0, 0], [0, 0, 0, 0], -1]

        # generate random block
        self.update_curr_block(random.choice(all_blocks))

        # generate 3 random next blocks
        self.block_pouch = self.fill_block_pouch()
        self.next_blocks = []
        for i in range(3):
            generated_block = self.generate_new_block(self.block_pouch)
            self.next_blocks.append(generated_block)

        # update state
        self.height = [0] * len(self.board[0])
        self.holes = [0] * len(self.board[0])
        self.bumpiness = [0] * (len(self.board[0]) - 1)
        self.rows_cleared = 0
        self.state = self.convert_to_state(self.block_type, self.hold, self.next_blocks, self.height, self.holes)

        # increment iterations and reset score and speed
        self.iterations += 1
        self.score = 0

        return np.array(self.state)


# get fitness score of genome
def eval_genome(genome, config):
    fitnesses = []
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # run each genome 5 times
    for runs in range(runs_per_net):
        env = TetrisEnv(display=False)
        observation = env.reset()
        fitness = 0.0
        done = False

        while not done:
            action = net.activate(observation)
            observation, reward, done, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness)
        # print high score if a high score is achieved
        global high_score
        if env.score > high_score:
            high_score = env.score
            print(f"High Score: {high_score}, Rows Cleared: {env.rows_cleared}")

    # genome fitness is determined by taking the mean of the score achieved in those 5 games
    return np.mean(fitnesses)


# evaluate all genomes in the population
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # load config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # return stats of each generation
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # determine the winner and save it in a pickle file
    winner = p.run(eval_genomes, 50)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


# test the ai
def test_ai():
    # load the model
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    # initiate environment
    env = TetrisEnv(display=True)
    episodes = 10
    env.iterations = 0

    # set up config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # let ai play 10 games
    for episode in range(1, episodes + 1):
        observation = env.reset()
        done = False
        env.score = 0
        reward_score = 0

        start_time = time.time()
        while not done:
            time.sleep(0.01)
            action = net.activate(observation)
            observation, reward, done, info = env.step(action)
            reward_score += reward

        # print score and the time it took
        print(f"Episode:{episode} Score:{env.score} Reward Score:{reward_score} Lines Cleared: {env.rows_cleared}")
        print(time.time() - start_time)

    pygame.quit()

if __name__ == '__main__':
    run()
    test_ai()
