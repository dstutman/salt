import numpy as np
import pandas as pd

skip = 3000

CSV = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\sverdrupxxl\LolaRDR_-90N-88N_175E235E_20210615T032611348_pts_csv.csv"

df = pd.read_csv(CSV)
df.columns = np.char.strip(np.array(df.columns, dtype="<U7"))

df["x"] = df["Pt_R"] * np.cos(np.radians(df["Pt_Lat"])) * np.cos(np.radians(df["Pt_Long"]))
df["y"] = df["Pt_R"] * np.cos(np.radians(df["Pt_Lat"])) * np.sin(np.radians(df["Pt_Long"]))
df["z"] = df["Pt_R"] * np.sin(np.radians(df["Pt_Lat"]))

x = df["x"][0::skip]
y = df["y"][0::skip]
z = df["z"][0::skip]
long = df["Pt_Long"][0::skip]
lat = df["Pt_Lat"][0::skip]
R = df["Pt_R"][0::skip]

size = np.size(x)
shrink_df = pd.concat([x,y,z,long,lat,R], axis=1)
shrink_df.to_csv("lolatopo_large{size}".format(size=str(size)))