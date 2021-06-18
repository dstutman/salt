import numpy as np
import pandas as pd

skip = 3000

CSV = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\sverdrupxl\lolatopo_lat_175.3_226.7_long_-90.0_-85.5"
df = pd.read_csv(CSV)

long = np.array([np.min(df["Pt_Long"]), np.max(df["Pt_Long"])])
lat = np.array([np.min(df["Pt_Lat"]), np.max(df["Pt_Lat"])])

print("Long = {0:f} - {1:f}\n"
      "Lat  = {2:f} - {3:f}".format(long[0], long[1], lat[0], lat[1]))
longmin = input("minimum longitude: ")
longmax = input("maximum longitude: ")
latmin = input("minimum latitude: ")
latmax = input("maximum latitude: ")

df = df[df.Pt_Long > float(longmin)]
df = df[df.Pt_Long < float(longmax)]
df = df[df.Pt_Lat > float(latmin)]
df = df[df.Pt_Lat < float(latmax)]

df.to_csv("lolatopo_lat_{0:.1f}_{1:.1f}_long_{2:.1f}_{3:.1f}".format(long[0], long[1], lat[0], lat[1]))