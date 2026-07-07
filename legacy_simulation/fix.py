import numpy as np
import pandas as pd


RAW_COLUMNS = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "proto", "state"]


dummy_df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, "tcp", "CON"]], columns=RAW_COLUMNS)
encoded_df = pd.get_dummies(dummy_df, columns=["proto", "state"], drop_first=True)

np.save("encoded_columns.npy", encoded_df.columns.tolist())

print("encoded_columns.npy has been created in this folder.")
