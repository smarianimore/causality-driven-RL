import pandas as pd
from path_repo import GLOBAL_PATH_REPO
from navigation.causality_algos import CausalDiscovery

df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/navigation/df_random_mdp_100000.pkl')
agent_n = 0

agent_X_columns = [s for s in df.columns.to_list() if f'agent_{agent_n}' in s]

df = df.loc[:, agent_X_columns]
print(df)
cd = CausalDiscovery(df, f'causal_knowledge/offline/navigation', f'agent{agent_n}_mdp_complete')
cd.training()
