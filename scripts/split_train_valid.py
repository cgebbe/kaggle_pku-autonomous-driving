import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path_folder',
                    default='../../data',
                    )
args = parser.parse_args()
path_folder = args.path_folder

# read csv
path_csv = os.path.join(path_folder, 'train.csv')
assert os.path.exists(path_csv), "No file at {}".format(path_csv)
df_org = pd.read_csv(path_csv, sep=',')

# split
df_train, df_valid = train_test_split(df_org, test_size=0.01, random_state=42)

# write split csv
df_train.to_csv(os.path.join(path_folder, 'train_train.csv'), sep=',', index=False)
df_valid.to_csv(os.path.join(path_folder, 'train_valid.csv'), sep=',', index=False)
print("=== Finished")
