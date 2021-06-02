import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
import seaborn as sns


data = pd.read_csv('Data_new_encoding.csv', sep=',', skiprows=38)

def teleview(lat, A,x, an):
    return lat-A*np.sin(x*(pi/180) + an + 152)

ra_of_asc_node = 125.08
lat = -86.5

fig, ax = plt.subplots()

ra_sim = np.linspace(0,360,1000)
lat_sim = teleview(lat, 16.56, ra_sim, ra_of_asc_node*(pi/180))
ra_sim[np.where(lat_sim < -90)] = ra_sim[np.where(lat_sim < -90)] + 180
lat_sim[np.where(lat_sim < -90)] = -lat_sim[np.where(lat_sim < -90)] - 180
ra_sim[np.where(ra_sim > 360)] = ra_sim[np.where(ra_sim > 360)] - 360
ra_sim[np.where(ra_sim < 0)] = ra_sim[np.where(ra_sim < 0)] + 360
galactic = np.zeros(1000)

galactic_angle = np.linspace(0,360,1000)
for i in range(1000):

    galactic[i] = 60*cos(galactic_angle[i]*pi/180)

sns.set_theme()
plt.xticks(np.linspace(0,360,6))
plt.yticks(np.linspace(-90,90,5))
ax.set_ylim(-90,90)
ax.set_xlim(0,360)
ax.set_facecolor("k")
hlines = np.linspace(-90,90,5)
vlines = np.linspace(0,360,6)
# grid
plt.hlines(hlines, 0,360, color= "gray", linestyles="--",zorder=0, linewidths=0.5)
plt.vlines(vlines, -90, 90, color= "gray", linestyles="--",zorder=0, linewidths=0.5)

#tele
plt.scatter(ra_sim, lat_sim, c = "r", s = 0.1, linewidth=3)
# exo
plt.scatter(data['ra'], data['dec'], s=1.5, marker="*",
            c="white", alpha=0.6, label=None)
#galaxy
plt.plot(galactic_angle, galactic, linestyle='--', label="Galactic plane", linewidth=3)
# dummy
plt.plot((-100, -90), (-180, -170),c = "r", label="Telescope observable path", linewidth=3)
plt.scatter(-100, -180, marker="*",
            c="white",label="Confirmed exoplanets")
plt.ylabel("Declination [deg]")
plt.xlabel("Right Ascension [deg]")
plt.title("Observable region")
plt.legend()
plt.show()
