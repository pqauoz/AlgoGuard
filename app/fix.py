import numpy as np
import pandas as pd

# 1. Create a dummy dataframe with the exact 10 raw column names
cols = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'proto', 'state']
dummy_df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 'tcp', 'CON']], columns=cols)

# 2. Apply the exact same One-Hot Encoding used in training
encoded_df = pd.get_dummies(dummy_df, columns=['proto', 'state'], drop_first=True)

# 3. Save the 145 column names directly into the current (app) folder
np.save('encoded_columns.npy', encoded_df.columns.tolist())

print("✅ Success! encoded_columns.npy has been created in this folder.")