import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rewards(values, name):
    fig = plt.figure(dpi=500)
    plt.title(f'{name} - {len(values)}')
    x = np.arange(0, len(values), 1)
    plt.plot(x, values)

    plt.show()


def discretize(df: pd.DataFrame, n_bins: int):
    pass


df_no = pd.read_pickle(
    'C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\causal_knowledge\\navigation_agent_3\df_original.pkl')

col_reward = [s for s in df_no.columns.to_list() if 'reward' in s]
plot_rewards(df_no[col_reward], col_reward)

col_action = [s for s in df_no.columns.to_list() if 'action' in s]
plot_rewards(df_no[col_reward], col_action)
