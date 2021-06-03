import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as cm
import math
import scipy as sc

s = np.sign

Tmax = 10

m = 450
J = 450

l1 = 1.2
l2 = 0.0
l3 = -1.2

k1 = 1/3
k2 = 1/3
# k2 = 0
k3 = 1/3

c1 = 450/3
c2 = 450/3
# c2 = 0
c3 = 450/3

s1 = s(l1)
s2 = s(l2)
s3 = s(l3)

print(f"s1 = {s1}, s2 = {s2}, s3 = {s3},")


# x = [x1, x2, theta theta2] = [x, xdot, theta, thetadot]

A = np.array([[0, 1, 0, 0],
              [-(k1+k2+k3)/m, -(c1+c2+c3)/m, (k1*l1 + k2*l2 + k3*l3)/m, (c1*l1 + c2*l2 + c3*l3)/m],
              [0, 0, 0, 1],
              [(s1*k1*l1 + s2*k2*l2 + s3*k3*l3)/J, (s1*c1*l1 + s2*c2*l2 + s3*c3*l3)/J, -(s1*k1*l1*l1 + s2*k2*l2*l2 + s3*k3*l3*l3)/J, -(s1*c1*l1*l1 + s2*c2*l2*l2 + s3*c3*l3*l3)/J]])

B = np.array([[0,0,0,0,0,0],
              [c1/m, c2/m, c3/m, k1/m, k2/m, k3/m],
              [0,0,0,0,0,0],
              [-(s1*c1*l1)/J, -(s2*c2*l2)/J, -(s3*c3*l3)/J, -(s1*k1*l1)/J, -(s2*k2*l2)/J, -(s3*k3*l3)/J]])

# u = [z1dot, z2dot, z3dot, z1, z2, z3]

C = np.array([[1 ,0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0], 
              [0, 0, 0, 1]])

D = np.array([[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

def wave(t, phase=0):
    maxAmplitude = 0.02 # [m]
    baseFreq = 21
    baseFreq *= 2 * math.pi
    baseFreq = 8
    wave = maxAmplitude * np.sin(t*baseFreq)
    wave_dot = maxAmplitude * baseFreq * np.cos(t*baseFreq + phase)

    return np.array([wave, wave_dot])

T = np.arange(0, Tmax, 0.1)

U = np.concatenate((wave(T), wave(T), wave(T)), axis=0)
# print((U.T[0]))

ss = cm.StateSpace(A, B, C, D)
TT, yout, xout = cm.forced_response(ss, T=T, X0=[0, 0, 0, 0], U=U, return_x=True)

dis = list(yout[0])
vel = list(yout[1])
angdis = list(np.degrees(yout[2]))
angvel = list(np.degrees(yout[3]))


fig, axs = plt.subplots(2, 2)

fig.suptitle(f"m={round(m, 3)},k={round(k1, 3)}, c={round(c1, 3)}, c/m={round(c1/m,3)}")

axs[0, 0].plot(T, dis, label="displacement")
axs[0, 0].plot(T, U[0], label="moonquake displacement", linestyle="--")


axs[0, 1].plot(T, vel, label="velocity")
axs[0, 1].plot(T, wave(T)[1], label="moonquake velocity", linestyle="--")

axs[1, 0].plot(T, angdis, label="ang displacement")
axs[1, 1].plot(T, angvel, label="ang velocity")

axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()

axs[0, 0].grid()
axs[0, 1].grid()
axs[1, 0].grid()
axs[1, 1].grid()


fig.tight_layout()

plt.show()
