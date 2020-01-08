import pandas as pd
from sklearn.model_selection import train_test_split

path_csv = r"../data/train.csv"
df_org = pd.read_csv(path_csv, sep=',')

df_train, df_valid = train_test_split(df_org, test_size=0.01, random_state=42)
df_train.to_csv('../data/train_train.csv', sep=',', index=False)
df_valid.to_csv('../data/train_valid.csv', sep=',', index=False)
print("=== Finished")

if False:
    regr_dict = {'x': 0,
                 'y': 0,
                 'z': 0,
                 'yaw': 0,
                 'pitch_sin': 0,
                 'pitch_cos': 0,
                 'roll': 0,
                 }
    # sorted(regr_dict) ->  ['pitch_cos', 'pitch_sin', 'roll', 'x', 'y', 'yaw', 'z']
