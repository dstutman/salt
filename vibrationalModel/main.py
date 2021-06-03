import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as cm
import math
import scipy as sc

# Wave function
def wave(t):
    baseFreq = 8  # radians/second
    maxAmplitude = 0.02

    wave = maxAmplitude * np.sin(t*baseFreq)
    wave_dot = maxAmplitude * baseFreq * np.cos(t*baseFreq)

    return np.array([wave, wave_dot])


# Constant values
m = 450
ks = 0.2
damp = m
Tmax = 10
dc = damp/m

# Laplace variable
s = cm.tf([1, 0], [1])

# Input equation x1 = xm, x2 = xm_dot, u1 = r, u2 = r_dot

A = np.array([[0, 1],
              [-ks/m, -dc]])

B = np.array([[0, 0],
              [ks/m, dc]])

C = np.array([[1, 0,], [0, 1,]])
D = np.array([[0, 0], [0, 0]])


T = np.arange(0,Tmax,0.05)

U = wave(T)

ss = cm.StateSpace(A, B, C, D)

_, yout, xout = cm.forced_response(ss, T=T, X0=[0,0], U = U, return_x=True)


dis = list(yout[0])
vel = list(yout[1])

fig, axs = plt.subplots(2, 1)

fig.suptitle(f"m={m},k={ks}, c={damp}, c/m={round(dc,3)}")



axs[0].plot(T, dis, label="displacement")
axs[0].plot(T, U[0], label="moonquake displacement", linestyle="--")
axs[0]

axs[1].plot(T, vel, label="velocity")
axs[1].plot(T, wave(T)[1], label="moonquake velocity", linestyle="--")

axs[0].legend()
axs[1].legend()

axs[0].grid()
axs[1].grid()

fig.tight_layout()

plt.show()

