import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as cm
import math
import scipy as sc
from matplotlib.animation import FuncAnimation
from tipfinder import tipfinder, getFreq

s = np.sign

Tmax = 1
stopLinear = Tmax
stopToZero = Tmax
randomFreq=0
timestep = 0.001 # do not change
bodyWidth = 1.5
initialRotation = math.radians(0)
initialDisplacement = 0.0

debug = False
plot = True
animate = True
save = False

maxAmplitude = 0.02  # [m]
# baseFreq = 0.1

m = 1750
J = 125 # = 56.55cos(11deg) + 70.3cos(11deg) (ixx+iyy)

l1 = 0.9
l2 = 0
l3 = -l1

kRover = 1000000000

# k2 must be less than 40*k1 or 40*k2 
k1 = 6000*2
k2 = 6000*2
k3 = 6000*2

c1 = m * 0.95 * 3
c2 = m * 0.95 * 3
c3 = m * 0.95 * 3

print(f"c1,c2,c3 = {c1}")

print(f"Keq = {k1+k2+k3}")
print(f"Ceq = {1/((1/c1)+(1/c2)+(1/c3))}")
print(f"Ceq/m = {(1/((1/c1)+(1/c2)+(1/c3)))/m}")

maxBaseFreqHz = 21

baseFreqHz = min(np.sqrt(kRover/m), maxBaseFreqHz)
baseFreq = baseFreqHz * (2 * math.pi)

print(f"Spring natural freq = {np.sqrt(k1/m)}")

# x = [x1, x2, theta theta2] = [x, xdot, theta, thetadot]

A = np.array([[0, 1, 0, 0],
              [-(k1+k2+k3)/m, -(c1+c2+c3)/m, (k1*l1 + k2*l2 + k3*l3)/m, (c1*l1 + c2*l2 + c3*l3)/m],
              [0, 0, 0, 1],
              [(k1*l1 + k2*l2 + k3*l3)/J, (c1*l1 + c2*l2 + c3*l3)/J, -(k1*l1*l1 + k2*l2*l2 + k3*l3*l3)/J, -(c1*l1*l1 + c2*l2*l2 + c3*l3*l3)/J]])

B = np.array([[0,0,0,0,0,0],
              [c1/m, c2/m, c3/m, k1/m, k2/m, k3/m],
              [0,0,0,0,0,0],
              [-(c1*l1)/J, -(c2*l2)/J, -(c3*l3)/J, -(k1*l1)/J, -(k2*l2)/J, -(k3*l3)/J]])

# u = [z1dot, z2dot, z3dot, z1, z2, z3]

C = np.array([[1 ,0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0], 
              [0, 0, 0, 1]])

D = np.array([[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

if debug:
    print("Start Matrix unit test")
    print("A thetadotdot", A[3])
    print("B thetadotdot", B[3])
    print("End Matrix unit test")

def wave(t, phase=0, bump=0, stop=Tmax, stop2=Tmax):
    randomizer = randomFreq*(np.random.rand(len(t))*10-10)

    wave =  maxAmplitude * np.sin(t*(baseFreq+randomizer) + phase)
    wave_dot = maxAmplitude * (baseFreq+randomizer) * np.cos(t*(baseFreq+randomizer) + phase)

    wave = np.where(t < stop, wave, wave * ((Tmax-t)/Tmax))
    wave = np.where(t < int(stop2), wave, 0)

    wave_dot = np.where(t < stop, wave_dot, wave_dot * ((Tmax-t)/Tmax))
    wave_dot = np.where(t < int(stop2), wave_dot, 0)

    if debug:
        print(f"frequency = {baseFreq} [rad/s], phase = {phase} [rad], timestep = {Tmax/len(t)}")

    return np.array([wave+bump, wave_dot])

T = np.arange(0, Tmax, timestep)

phase1 = 0
phase2 = 0
phase3 = 0

bump1 = 0.0
bump2 = 0.0
bump3 = 0.0

waveW1 = wave(T, phase1, bump=bump1, stop=int(stopLinear), stop2=int(stopToZero)).tolist()
waveW2 = wave(T, phase2, bump=bump2, stop=int(stopLinear), stop2=int(stopToZero)).tolist()
waveW3 = wave(T, phase3, bump=bump3, stop=int(stopLinear), stop2=int(stopToZero)).tolist()

U = np.concatenate((waveW1, waveW2, waveW3), axis=0)

ss = cm.StateSpace(A, B, C, D)
cm.damp(ss)

TT, yout, xout = cm.forced_response(ss, T=T, X0=[initialDisplacement, 0, initialRotation, 0], U=U, return_x=True)

dis = list(yout[0])
vel = list(yout[1])
accel = (np.gradient(vel)/timestep)
angdis = list(np.round(np.degrees(yout[2]), 15))
angvel = list(np.round(np.degrees(yout[3]), 15))
angAccell = (np.gradient(angvel)/timestep)
surfaceAccel = (np.gradient(waveW1[1])/timestep)
angvelRads = list(np.round(yout[3], 15))
angAccelRads = (np.gradient(angvelRads)/timestep)
totalAccelleration = -(angAccelRads * (bodyWidth/2)) + accel

# tips = tipfinder(accel[500:])
# calcFreq = getFreq(accel[500:])
# print(f"Freq = {round(calcFreq, 3)} Hz")
# print(f"Max accel = {round(max(accel), 3)}")

### Plotting
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6,7))
# accelFig, accelAx = plt.subplots(1, 1)

# fig.suptitle(f"K = 6000 f = ")
# fig.suptitle(f"m={round(m, 3)},k1={round(k1, 3)}, c1={round(c1, 3)}, c1/m={round(c1/m,3)}, freq={round(baseFreq, 3)}[rad/s]/{round(baseFreqHz, 3)}[Hz]")
fig.suptitle(f"Frequency {baseFreqHz} Hz")
axs[0].plot(T, waveW1[0], label="moonquake displacement", linestyle="--", color="orange")
axs[0].plot(T, dis, label="displacement", color="tab:blue")

axs[1].plot(T, waveW1[1], label="moonquake velocity", linestyle="--", color="orange")
axs[1].plot(T, vel, label="velocity", color="tab:blue")
axs[2].plot(T, surfaceAccel, label="moonquake acceleration", linestyle="--", color="orange")
axs[2].plot(T, accel, label="acceleration", color="tab:blue")
# axs[0, 1].plot(T[100::], (np.gradient(dis)/timestep)[100::], label="test vello") # verifying the acceleration graph

axs[2].set_xlabel("Time [s]")

axs[0].set_ylabel("Displacement [m]")
axs[1].set_ylabel("Velocity [m/s]")
axs[2].set_ylabel("Acceleration [m/s^2]")

axs[0].grid()
axs[1].grid()
axs[2].grid()

axs[0].legend(loc=4)
axs[1].legend(loc=4)
axs[2].legend(loc=4)

if plot:
    plt.show()
