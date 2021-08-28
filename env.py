import numpy as np


class Environment:
    def __init__(self, num_rows=4, num_cols=4):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.G = None
        self.grid = None
        self.reset_grid()

    def build_wall(self, mask, filled=1):
        self.grid = np.ma.array(self.grid, mask=mask).filled(filled)

    def reset_grid(self):
        grid = np.zeros((self.num_rows, self.num_cols))
        self.grid = grid

    def update_robot(self):
        self.grid[self.robot] = 2

    def is_inside_grid(self, row, col):
        if row < 0 or row > 3 or col < 0 or col > 3:
            return False
        else:
            return True

    def is_not_wall(self, row, col):
        if self.grid[row, col] == 1:
            return False
        else:
            return True

    def is_allowed_move(self, row, col):
        if not self.is_inside_grid(row, col):
            return False
        elif not self.is_not_wall(row, col):
            return False
        elif row == 0 and col == 0:
            return False
        else:
            return True

    def update_position(self, row, col):
        if self.is_allowed_move(row, col):
            self.grid[self.robot] = 0
            self.robot = (row, col)
            self.grid[self.robot] = 2

    def is_game_over(self):
        row, col = self.robot
        if row == 3 and col == 3:
            return True
        else:
            return False
