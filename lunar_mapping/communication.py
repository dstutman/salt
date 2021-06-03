import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7,4))
ax = Axes3D(fig)
ax.set_facecolor("k")
plt.axis('off')

observationstation = np.array([-47, -30,-1.5])
sverdrupstation = np.array([-11.41, -11.02, 2.5])
malapertstation = np.array([120, 6.3, 1.5])
observationpowerstation = np.array([-59.7, -14.4, 1.6])
earthspherical = np.array([0,0,1000])

# Celestial bodies | Moon
R = 1737.1 # [km]
u, v = np.mgrid[1.05*np.pi:0.16*np.pi:20j, 0.004*np.pi:0.03*np.pi:8.5j]
x = R * np.cos(u)*np.sin(v)
y = R * np.sin(u)*np.sin(v)
z = R * np.cos(v) - R
ax.plot_wireframe(x, y, z, color="gray", linewidth=0.5, zorder=-1)
u, v = np.mgrid[-0.7*np.pi:-0.03*np.pi:15.06j, 0:0.03*np.pi:10j]
x = R * np.cos(u)*np.sin(v)
y = R * np.sin(u)*np.sin(v)
z = R * np.cos(v) - R
ax.plot_wireframe(x, y, z, color="gray", linewidth=0.5, zorder=-1)
u, v = np.mgrid[1.05*np.pi:1.3*np.pi:5.6j, 0.016*np.pi:0.03*np.pi:5.5j]
x = R * np.cos(u)*np.sin(v)
y = R * np.sin(u)*np.sin(v)
z = R * np.cos(v) - R
ax.plot_wireframe(x, y, z, color="gray", linewidth=0.5, zorder=-1)
u, v = np.mgrid[-0.03*np.pi:0.16*np.pi:5.6j, 0:0.018*np.pi:7j]
x = R * np.cos(u)*np.sin(v)
y = R * np.sin(u)*np.sin(v)
z = R * np.cos(v) - R
ax.plot_wireframe(x, y, z, color="gray", linewidth=0.5, zorder=-1)

earthcartisian = np.array([earthspherical[2]*np.cos(earthspherical[0])*np.cos(earthspherical[1]),
                           earthspherical[2]*np.cos(earthspherical[0])*np.sin(earthspherical[1]),
                           earthspherical[2]*np.sin(earthspherical[0])-20])

# Celestial bodies | Earth
R_E = 100 #6371
u, v = np.mgrid[-np.pi:np.pi:20j, 0:np.pi:10j]
x = R_E * np.cos(u)*np.sin(v)
y = R_E * np.sin(u)*np.sin(v)
z = R_E * np.cos(v)*0.5
ax.plot_wireframe(x+earthcartisian[0], y+earthcartisian[1], z+earthcartisian[2],
                  color="royalblue", linewidth=0.5, zorder=-1)

# Topology | sverdrup crater
csv = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\lolawidewide\lolatopo_large2205"
dfsverdrupcrater = pd.read_csv(csv)
xsc = dfsverdrupcrater["x"]
ysc = dfsverdrupcrater["y"]
zsc = -dfsverdrupcrater["z"] - R
ax.plot_trisurf(xsc, ysc, zsc, cmap="coolwarm",zsort="min",zorder=-1)

# Topology | sverdrup rim
csv = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\lolaRDR_largelarge\lolatopo_lat_189.3_234.9_long_-89.5_-89.0"
dfsverdruprim = pd.read_csv(csv)
xsr = dfsverdruprim["x"]
ysr = dfsverdruprim["y"]
zsr = -dfsverdruprim["z"] - R
ax.plot_trisurf(xsr, ysr, zsr, cmap="coolwarm",zsort="min",zorder=-1)


# Topology | shackleton
csv = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\lolashack\lolatopo_large834"
dfshackleton = pd.read_csv(csv)
xs = dfshackleton["x"]
ys = dfshackleton["y"]
zs = -dfshackleton["z"] - R
ax.plot_trisurf(xs, ys, zs, cmap="coolwarm",zsort="min",zorder=-1)

# Topology | malapert
csv = r"C:\Users\noutb\PycharmProjects\LR\bsc-y3-q4\DSE\data\malapert\lolatopo_large3223"
dfmalapert = pd.read_csv(csv)
xm = dfmalapert["x"]
ym = dfmalapert["y"]
zm = -dfmalapert["z"] - R
ax.plot_trisurf(xm, ym, zm, cmap="coolwarm",zsort="min",zorder=-1)

min, max = np.min((np.min(y), np.min(x))), np.max((np.max(y), np.max(x)))
ax.set_aspect("equal")
ax.set_xlim3d(min, 200) # max)
ax.set_ylim3d(min, 200) # max)
ax.set_zlim3d(np.min(z)-30,np.max(z)+30)

# Communication | observation to sverdrup
comm_observation_sverdrup = np.vstack((observationstation, sverdrupstation))
ax.plot(comm_observation_sverdrup[:,0],
        comm_observation_sverdrup[:,1],
        comm_observation_sverdrup[:,2], c="w", ls="--", zorder=12)

# Communication | sverdrup to malapert
comm_sverdrup_malapert = np.vstack((sverdrupstation, malapertstation))
ax.plot(comm_sverdrup_malapert[:,0],
        comm_sverdrup_malapert[:,1],
        comm_sverdrup_malapert[:,2], c="w", ls="--", zorder=8)

# Communication | malapert to earth
comm_malapert_earth = np.vstack((malapertstation, earthcartisian))
ax.plot(comm_malapert_earth[:,0],
        comm_malapert_earth[:,1],
        comm_malapert_earth[:,2], c="w", ls="--", zorder=9)

# Power | power station to observation station
pow_ob = np.vstack((observationstation, observationpowerstation))
ax.plot(pow_ob[:,0],
        pow_ob[:,1],
        pow_ob[:,2], c="r", ls="--", zorder=9)


# Communication | station markers
ax.scatter(sverdrupstation[0], sverdrupstation[1], sverdrupstation[2],
           c="r", marker="^", zorder=13,
           label="Sverdrup Relay Station")

ax.scatter(observationstation[0], observationstation[1], observationstation[2],
           c="g", marker='^', zorder=11,
           label="Observation Station", depthshade=False)

ax.scatter(malapertstation[0], malapertstation[1], malapertstation[2],
           c="b", marker='^', zorder=13,
           label="Observation Station", depthshade=False)

ax.scatter(earthcartisian[0], earthcartisian[1], earthcartisian[2],
           c="c", marker="^", zorder=13,
           label="Earth Receiving Station", depthshade=False)
ax.scatter(observationpowerstation[0], observationpowerstation[1], observationpowerstation[2],
           c="y", marker="x", zorder=13,
           label="Power Station for Observation Station")


o_s = np.sqrt(np.sum((observationstation-sverdrupstation)**2))
s_m = np.sqrt(np.sum((sverdrupstation-malapertstation)**2))
m_e = np.sqrt(384400**2 + R**2)
op_o = np.sqrt(np.sum((observationpowerstation-observationstation)**2))

ax.text(sverdrupstation[0], sverdrupstation[1], sverdrupstation[2]+4,
        "Relay line {0:.1f} [km]".format(o_s), color='w', zorder=13)
ax.text(malapertstation[0], malapertstation[1], malapertstation[2]+4,
        "Communication line {0:.1f} [km]".format(s_m), color='w', zorder=13)
ax.text(earthcartisian[0], earthcartisian[1], earthcartisian[2],
        "Earth communication line ≈ {0:.1f} [km]".format(m_e), color="w", zorder=13)
ax.text(observationpowerstation[0], observationpowerstation[1], observationpowerstation[2]+6,
        "Power for observation station {0:.1f} [km]".format(op_o), color="w", zorder=13)

ax.legend()

print("Communication lines | Observation to Sverdrup rim: {0:.1f}       [km] \n"
      "                      Sverdrup rim to Malapert:    {1:.1f}      [km] \n"
      "                      Malapert to Earth:           {2:.1f}   [km] \n"
      "Power lines         | Power for Observation        {3:.1f}       [km]".format(o_s,
                                                                          s_m,
                                                                          np.sqrt(384400**2 + R**2),
                                                                          op_o))
plt.show()