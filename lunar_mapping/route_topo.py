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

csv1 = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\sverdrupxl\lolatopo_lat_175.3_226.7_long_-90.0_-85.5"
csv2 = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\lolashack\lolatopo_large834"

sver = pd.read_csv(csv1)
shack = pd.read_csv(csv2)

topo = pd.concat([sver,shack],ignore_index=True)

x = topo["x"]
y = topo["y"]
z = -topo["z"] - R
ax.plot_trisurf(x,y,z, cmap="coolwarm", linewidth=0.05 ,zorder=-1)

# min, max = np.min((np.min(y), np.min(x))), np.max((np.max(y), np.max(x)))
# ax.set_aspect("equal")
# ax.set_xlim3d(min, max)
# ax.set_ylim3d(min, max)
#ax.set_zlim3d(np.min(z)-1,np.max(z)+1)

xs = shack["x"]
ys = shack["y"]
zs = -shack["z"] - R
shackbase = np.array([[np.mean(xs)],
                      [np.mean(ys)],
                      [z[np.where((xs <= np.mean(xs)+0.5) &
                                  (xs >= np.mean(xs)-0.5) &
                                  (ys <= np.mean(ys)+0.5) &
                                  (ys >= np.mean(ys)-0.5))[0][0]]]])

ax.scatter3D(shackbase[0], shackbase[1], shackbase[2]+2,
           c="r", marker='^', zorder=2, s=15,
           label="Shackleton crater", depthshade=False)

route = np.zeros(15, dtype=object)
minlong = 0
maxlong = 360
loc = shackbase
for i in range(np.size(route)):
      # stepz = np.array(z[np.where((topo.Pt_Long <=maxlong) &
      #                             (topo.Pt_Long >= minlong) &
      #                             (topo.x < float(loc[0])))[0]])
      # stepx = np.array(x[np.where((topo.Pt_Long <=maxlong) &
      #                             (topo.Pt_Long >= minlong) &
      #                             (topo.x < float(loc[0])))[0]])
      # stepy = np.array(y[np.where((topo.Pt_Long <=maxlong) &
      #                             (topo.Pt_Long >= minlong) &
      #                             (topo.x < float(loc[0])))[0]])
      if loc[2] > 1.2:
          stepz = np.array(z[np.where((topo.y < float(loc[1])) &
                                      (topo.x < float(loc[0]))-0)[0]])
          stepx = np.array(x[np.where((topo.y < float(loc[1])) &
                                      (topo.x < float(loc[0]))-0)[0]])
          stepy = np.array(y[np.where((topo.y < float(loc[1])) &
                                      (topo.x < float(loc[0]))-0)[0]])
      else:
          stepz = np.array(z[np.where((topo.y < float(loc[1])) &
                                      (topo.x < float(loc[0])))[0]])
          stepx = np.array(x[np.where((topo.y < float(loc[1])) &
                                      (topo.x < float(loc[0])))[0]])
          stepy = np.array(y[np.where((topo.y < float(loc[1])) &
                                      (topo.x < float(loc[0])))[0]])

      flatdistance = np.sqrt((loc[0]-stepx)**2 + (loc[0]-stepy)**2)

      fov = zip(flatdistance, stepx, stepy, stepz)
      fov = np.array(list(sorted(fov)))[:300]
      fov = fov.reshape(int(np.size(fov)/4), 4)
      flatdistance, stepx, stepy, stepz = fov[:,0], fov[:,1], fov[:,2], fov[:,3]
      slope = (stepz - shackbase[2])**2 / flatdistance
      loc = np.array([stepx[np.where(slope == np.min(slope))],
                      stepy[np.where(slope == np.min(slope))],
                      stepz[np.where(slope == np.min(slope))]])
      route[i] = loc
route = np.concatenate(route)
route = route.reshape(int(np.size(route)/3), 3)
ax.plot(route[:,0], route[:,1], route[:,2],
           c="k", marker='x', zorder=12,
           label="step")
#ax.set_zlim3d((-40,40))
ax.view_init(elev=90., azim=180)

plt.show()
#plt.plot(route[:,0], route[:,2])
#plt.show()