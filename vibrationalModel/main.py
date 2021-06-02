import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as cm
import math
import scipy as sc

# Wave function
def wave(t):
    # return np.sin(x/baseFreq) + np.random.normal(scale = 0.1, size = len(x))
    half = len(t)/2
    a = 1
    b = 0.2
    c = 0.3
    divc = 0

    baseFreq = 21/2  #HERTZZZZZZZZZZZZ
    baseFreq = 2 * baseFreq * math.pi  # radians/second
    maxAmplitude = 1/(a+b+c) * 0.02

    # wave = (a * np.sin(t*baseFreq) + b * np.sin(t*(baseFreq*0.23)) + c * np.sin(t*(baseFreq*2) + divc))
    # wave_dot = ((a*baseFreq) * np.cos(t*baseFreq) + (b*(baseFreq*0.23)) * np.cos(t*(baseFreq*0.23)) + (c*(baseFreq*2)) * np.cos(t*(baseFreq*2) + divc))
    # wave_dot = ((a*baseFreq) * np.cos(t*baseFreq) + (b*(baseFreq*0.23)) * np.cos(t*(baseFreq*0.23)) + (c*(baseFreq*2)) * np.cos(t*(baseFreq*2) + divc))

    baseFreq=baseFreq*2
    wave =  np.sin(t*baseFreq)
    wave_dot = baseFreq * np.cos(t*baseFreq)
    wave_dot_dot = baseFreq**2 * -np.sin(t*baseFreq)
    # wave[int(half)::] = 0
    # wave_dot[int(half)::] = 0
    return np.array([wave*maxAmplitude, wave_dot*maxAmplitude, wave_dot_dot*maxAmplitude])


# def wave(t):
#     return sc.signal.square(t)
# Constant values
m = 450/6
ks = 0.1
damp = 1
Tmax = 2000
# dc = damp/m
dc = 1
# Laplace variable
s = cm.tf([1, 0], [1])

# Input equation x1 = xm, x2 = xm_dot, u1 = r, u2 = r_dot

A = np.array([
              [0, 1],
              [-ks/m, -dc]])

B = np.array([
              [0, 0, 0],
              [ks/m, dc, 1]])

C = np.array([[1, 0,], [0, 1,]])
D = np.array([[0, 0, 0], [0, 0, 0]])


T = np.arange(0,Tmax,0.1)

U = wave(T)

ss = cm.StateSpace(A, B, C, D)
print(U.T[0])
print(U.T[0].T)
yout, xout = cm.forced_response(ss, T=T, X0=[0,0], U = U)

dis = list(xout[0])
vel = list(xout[1])
plt.plot(T, dis, label="displacement")
# plt.plot(T, vel, label="velocity")
plt.plot(T, U[0], label="moonquake", linestyle="--")
# plt.plot(T, wave(T)[1], label="moonquake velocity", linestyle="--")
plt.legend()
plt.show()

