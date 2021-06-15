import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns

observationstation = np.array([-47, -30,-3.2])
sverdrupstation = np.array([-11.41, -11.02, 2.5])
malapertstation = np.array([120, 6.3, 1.5])
observationpowerstation = np.array([-59.7, -14.4, 0.7])
earthspherical = np.array([0,0,1000])


R = 1737.1 # [km]

rcParams['xtick.color'] = 'white'
fig = plt.figure(figsize=(7,4))
ax = Axes3D(fig)
ax.set_facecolor("k")
sns.set_theme()
ax.w_xaxis.line.set_color("w")
ax.w_yaxis.line.set_color("w")
ax.w_zaxis.line.set_color("w")
ax.tick_params(axis='x', colors='w')
ax.tick_params(axis='y', colors='w')
ax.tick_params(axis='z', colors='w')
ax.yaxis.label.set_color('w')
ax.xaxis.label.set_color('w')
ax.zaxis.label.set_color("w")
ax.title.set_color("w")



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
sns.set_theme()
ax.plot_trisurf(xsr, ysr, zsr, cmap="coolwarm",zsort="min",zorder=-1, linewidth=0.05)
# print(xsr[np.where(zsr==np.max(zsr))[0]],
#       ysr[np.where(zsr==np.max(zsr))[0]],
#       zsr[np.where(zsr==np.max(zsr))[0]])


min, max = np.min((np.min(y), np.min(x))), np.max((np.max(y), np.max(x)))
ax.set_aspect("equal")
ax.set_xlim3d(min, max)
ax.set_ylim3d(min, max)
ax.set_zlim3d(np.min(z)-6,np.max(z)+6)
ax.plot_trisurf(x, y, z, cmap="coolwarm", linewidth=0.05 ,zorder=-1)

# Power | power station to observation station
pow_ob = np.vstack((observationstation, observationpowerstation))
ax.plot(pow_ob[:,0],
        pow_ob[:,1],
        pow_ob[:,2], c="r", ls="--", zorder=9, label="Power line")

# Communication | observation to sverdrup
comm_observation_sverdrup = np.vstack((observationstation, sverdrupstation))
ax.plot(comm_observation_sverdrup[:,0],
        comm_observation_sverdrup[:,1],
        comm_observation_sverdrup[:,2], c="w", ls="--", zorder=12, label="Comm. lines")

# Communication | station markers
ax.scatter(sverdrupstation[0], sverdrupstation[1], sverdrupstation[2],
           c="w", marker="^", zorder=13,
           label="Sverdrup Relay Station")

ax.scatter(observationstation[0], observationstation[1], observationstation[2],
           c="k", marker='^', zorder=13,
           label="Observation Station", depthshade=False)

ax.scatter(observationpowerstation[0], observationpowerstation[1], observationpowerstation[2],
           c="r", marker="x", zorder=13,
           label="Power Station for Observation Station")
# Communication | sverdrup to malapert
comm_sverdrup_malapert = np.vstack((sverdrupstation, malapertstation))
ax.plot(comm_sverdrup_malapert[:,0],
        comm_sverdrup_malapert[:,1],
        comm_sverdrup_malapert[:,2], c="w", ls="--", zorder=8)

o_s = np.sqrt(np.sum((observationstation-sverdrupstation)**2))
s_m = np.sqrt(np.sum((sverdrupstation-malapertstation)**2))
m_e = np.sqrt(384400**2 + R**2)
op_o = np.sqrt(np.sum((observationpowerstation-observationstation)**2))

ax.text(sverdrupstation[0]-10, sverdrupstation[1]-10, sverdrupstation[2]+0.5,
        "Relay line {0:.1f} [km]".format(o_s), color='w', zorder=13)
ax.text(observationpowerstation[0]-20, observationpowerstation[1]-20, observationpowerstation[2]+1,
        "Power for observation station {0:.1f} [km]".format(op_o), color="w", zorder=13)

ax.set_xlabel("Distance away from South pole [km]")
ax.set_ylabel("Distance away from South pole [km]")
ax.set_zlabel("Altitude [km]")
ax.set_title("Mission Topology SALT Observation Site", loc="left")

plt.legend()
plt.show()