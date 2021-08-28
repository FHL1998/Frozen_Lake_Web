import time
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import streamlit as st
from agent import Agent
from env import Environment

st.set_page_config(layout="wide")
st.title("Reinforcement Learning")
st.subheader("Solving gridworld through value iteration")
col1 = st.sidebar
col2, col3 = st.columns([4, 4])

num_rows = 4
num_cols = 4
initial_wall = [5, 7, 11, 12]
total_range = np.arange(num_rows*num_cols).reshape(num_rows,
                                                   num_cols
                                                   ).astype(np.float64)
col1.subheader("Build new wall")
col1.write("Add or delete a cell to change the wall")
select_wall = col1.multiselect("But make sure there is at least one path back home:)",
                               list((range(1, (num_rows*num_cols)-1))),
                               initial_wall
                               )

wall_row = [i // num_rows for i in select_wall]
wall_col = [i % num_rows for i in select_wall]
total_range[wall_row, wall_col] = np.nan

df = pd.DataFrame(total_range)
f, ax = plt.subplots(figsize=(3, 3))
mask = df.isnull()
ax = sns.heatmap(df,
                 mask=mask,
                 square=True,
                 linewidths=0.3,
                 vmin=0,
                 vmax=num_rows*num_cols,
                 cmap="RdBu_r",
                 annot=True,
                 cbar=False,
                 )
ax.set_facecolor("#9b95c9")
col1.pyplot(f)
col1.subheader("Change the number of training episodes")
col1.write("Longer episodes provide better results but take more time.")
episodes = col1.slider('Default value is good enough for most situations', 100, 300, 300)

col2.header("Initial State Value")
Environment = Environment(num_rows, num_cols)
Environment.build_wall(mask=mask)
agent = Agent(Environment)
f, ax = agent.pretraining_heatmap()
yticks = ["Start", "", "", "", ]
xticks = [ "", "", "", "Home"]
ax.set_xticklabels(xticks)
ax.set_yticklabels(yticks)
col2.pyplot(f)

if col2.button('Start Reinforcement Learning'):
    col2.subheader('Training in progress...')
    start = time.time()
    agent.learn(episodes=episodes)
    end = time.time()
    timetaken = end - start
    col2.write(f"Time taken: {timetaken}")

    if agent.succeeded:
        f2, ax2 = agent.post_training_heatmap()
        ax2.set_xticklabels(xticks)
        ax2.set_yticklabels(yticks)
        col3.header("Final State Value")
        col3.pyplot(f2)
        col3.subheader("Shortest path is shown in blue")
    else:
        col2.write("You might have block every path back home. Consider redoing your wall.")

del agent
del Environment
gc.collect()
