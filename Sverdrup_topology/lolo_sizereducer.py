import numpy as np
import pandas as pd

CSV = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\LolaRDR_-89N-88N_188E226E\LolaRDR_-89N-88N_188E226E_20210527T033750223_pts_csv.csv"

df = pd.read_csv(CSV)
df.columns = np.char.strip(np.array(df.columns, dtype="<U7"))

df["x"] = df["Pt_R"] * np.cos(np.radians(df["Pt_Lat"])) * np.cos(np.radians(df["Pt_Long"]))
df["y"] = df["Pt_R"] * np.cos(np.radians(df["Pt_Lat"])) * np.sin(np.radians(df["Pt_Long"]))
df["z"] = df["Pt_R"] * np.sin(np.radians(df["Pt_Lat"]))

x = df["x"][0::1500]
y = df["y"][0::1500]
z = df["z"][0::1500]
long = df["Pt_Long"][0::1500]
lat = df["Pt_Lat"][0::1500]
R = df["Pt_R"][0::1500]
shrink_df = pd.concat([x,y,z,long,lat,R], axis=1)
shrink_df.to_csv("lolatopo_1500")