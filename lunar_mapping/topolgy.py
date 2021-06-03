import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

R = 1737.1 # [km]

fig = plt.figure(figsize=(7,4))
ax = Axes3D(fig)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

csv = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\lolawidewide\lolatopo_large2205"

df = pd.read_csv(csv)
df.columns = np.char.strip(np.array(df.columns, dtype="<U7"))
x = df["x"]
y = df["y"]
z = -df["z"] - R

# Topology | sverdrup rim
csv = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\lolaRDR_largelarge\lolatopo_lat_189.3_234.9_long_-89.5_-89.0"
dfsverdruprim = pd.read_csv(csv)
xsr = dfsverdruprim["x"]
ysr = dfsverdruprim["y"]
zsr = -dfsverdruprim["z"] - R
ax.plot_trisurf(xsr, ysr, zsr, cmap="coolwarm",zsort="min",zorder=-1)
print(xsr[np.where(zsr==np.max(zsr))[0]],
      ysr[np.where(zsr==np.max(zsr))[0]],
      zsr[np.where(zsr==np.max(zsr))[0]])


min, max = np.min((np.min(y), np.min(x))), np.max((np.max(y), np.max(x)))
ax.set_aspect("equal")
ax.set_xlim3d(min, max)
ax.set_ylim3d(min, max)
ax.set_zlim3d(np.min(z)-6,np.max(z)+6)
ax.plot_trisurf(x, y, z, cmap="coolwarm",zsort="min")

plt.show()