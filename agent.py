import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


class Agent:
    def __init__(self, Environment):
        self.action_space = ["up", "down", "left", "right"]
        self.Environment = Environment
        self.G = None
        self.random_threshold = 0.9
        self.alpha = 0.9
        self.init_G()
        self.succeeded = False

    def init_G(self):
        self.G = np.random.random((self.Environment.num_rows,self.Environment.num_cols)) * -1
        self.G[0, 0] = 0
        self.G[self.Environment.num_rows-1, self.Environment.num_cols-1] = 0
        for row in range(self.Environment.num_rows):
            for col in range(self.Environment.num_cols):
                if self.Environment.grid[row, col] == 1.0:
                    self.G[row, col] = np.nan
        self.G = np.around(self.G, 2)

    def action_to_position(self, action):
        row, col = self.Environment.robot
        if action == "up":
            row = row-1
        elif action == "down":
            row = row+1
        elif action == "left":
            col = col-1
        elif action == "right":
            col = col+1
        else:
            raise Exception("No such action,please check again")
        return row, col

    def chooseRandomAction(self):
        while True:
            action = np.random.choice(self.action_space)
            row, col = self.action_to_position(action)
            if self.Environment.is_allowed_move(row, col):
                return action

    def chooseBestAction(self):
        payoff = []
        for action in self.action_space:
            row, col = self.action_to_position(action)
            if self.Environment.is_allowed_move(row, col):
                payoff.append(self.G[row, col])
            else:
                payoff.append(np.nan)
        action = np.nanargmax(payoff)
        return self.action_space[action]

    def BestActionPlusRandom(self):
        if np.random.random_sample() > self.random_threshold:
            action = self.chooseBestAction()
        else:
            action = self.chooseRandomAction()
        position = self.action_to_position(action)
        return position

    def update_G(self):
        memory = self.memory.copy()
        memory_reverse = memory[::-1]
        rewards = self.rewards.copy()
        rewards_reverse = rewards[::-1]
        target = 0
        for idx, state in enumerate(memory_reverse):
            target += rewards_reverse[idx]
            self.G[state] += self.alpha*(target - self.G[state])

    def learn(self, episodes=300, max_count=300):
        self.init_G()
        divider = episodes//10
        for episode in range(episodes):
            if episode % divider == 0:
                self.random_threshold *= 0.9
                self.alpha *= 0.9
            self.Environment.robot = (0, 0)
            self.memory = [(0, 0)]
            self.rewards = [0.0]
            count = 0
            while not self.Environment.is_game_over():
                count += 1
                self.Environment.grid[self.Environment.robot] = 0
                self.Environment.robot = self.BestActionPlusRandom()
                self.Environment.grid[self.Environment.robot] = 2
                self.memory.append(tuple(self.Environment.robot))
                reward = 0 if self.Environment.is_game_over() else -1
                if self.Environment.is_game_over():
                    self.succeeded = True
                self.rewards.append(reward)

                if count >= max_count:
                    break
            self.update_G()
        self.G = np.around(self.G, 2)

    def pretraining_heatmap(self):
        f, ax = plt.subplots(figsize=(3, 3))
        df = pd.DataFrame(self.G)
        df = df.round(3)
        df[df < -99] = -99
        mask = df.isnull()
        ax = sns.heatmap(df,
                         mask=mask,
                         square=True,
                         linewidths=0.3,
                         vmin=-1,
                         vmax=0,
                         cmap="RdBu_r",
                         annot=True,
                         cbar=False,
                         )

        ax.set_facecolor("#401c44")
        return f, ax

    def post_training_heatmap(self):
        f, ax = plt.subplots(figsize=(3, 3))
        df = pd.DataFrame(self.G)
        df = df.round(0)
        df[df < -99] = -99
        mask = df.isnull()
        ax = sns.heatmap(df,
                         mask=mask,
                         square=True,
                         linewidths=0.3,
                         vmin=-20,
                         vmax=0,
                         cmap="RdBu_r",
                         annot=True,
                         cbar=False,
                         )
        ax.set_facecolor("#401c44")
        best_path = self.get_shortest_path()
        for rect in best_path:
            ax.add_patch(
                         Rectangle(rect, 1, 1,
                                   fill=False,
                                   edgecolor='blue',
                                   lw=3
                                   )
                         )

        return f, ax

    def get_shortest_path(self):
        best_path = [(0, 0)]
        self.Environment.robot = (0, 0)
        row, col = self.Environment.robot
        while (row, col) != (3, 3):
            action = self.chooseBestAction()
            position = self.action_to_position(action)
            best_path.append(position)
            self.Environment.robot = position
            row, col = self.Environment.robot
        best_path = [(j, i) for i, j in best_path]
        return best_path
