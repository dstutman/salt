import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns



R = 1737.1 # [km]

fig = plt.figure(figsize=(7,4))
ax = Axes3D(fig)
sns.set_theme()

csv = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\sverdrupxxl\lolatopo_large2804"

sver = pd.read_csv(csv)
x = sver["x"]
y = -sver["y"]
z = -sver["z"] - R
# min, max = np.min((np.min(y), np.min(x))), np.max((np.max(y), np.max(x)))
# ax.set_aspect("equal")
# ax.set_xlim3d(min, max)
# ax.set_ylim3d(min, max)
ax.set_zlim3d(np.min(z)-1,np.max(z)+1)
ax.plot_trisurf(x, y, z, cmap="coolwarm", linewidth=0.05 ,zorder=-1)
print(np.min(sver["Pt_Long"]), np.max(sver["Pt_Long"]),
      np.min(sver["Pt_Lat"]), np.max(sver["Pt_Lat"]))

csv = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\lolashack\lolatopo_large834"

shack = pd.read_csv(csv)
x = shack["x"]
y = -shack["y"]
z = -shack["z"] - R
ax.plot_trisurf(x,y,z, cmap="coolwarm", linewidth=0.05 ,zorder=-1)

ax.view_init(elev=90., azim=180)
plt.show()