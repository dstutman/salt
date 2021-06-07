import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as cm
import math
import scipy as sc
from matplotlib.animation import FuncAnimation
from tqdm import tqdm as tqdm
from tipfinder import tipfinder, getFreq


s = np.sign

Tmax = 40
stopLinear = Tmax
stopToZero = Tmax
randomFreq = 0
timestep = 0.001 # do not change
bodyWidth = 1.5
initialRotation = math.radians(0)
initialDisplacement = 0.02

debug = False
plot = True
animate = True
save = False

maxAmplitude = 0.02  # [m]
# baseFreq = 0.1

m = 1500/2
J = 125 # = 56.55cos(11deg) + 70.3cos(11deg) (ixx+iyy)

l1 = 0.85
l2 = 0
l3 = -l1

kRover = 1000000000

# k2 must be less than 40*k1 or 40*k2 

c1 = 0.99*m
c2 = 0.99*m
c3 = 0.99*m

# print(f"Keq = {k1+k2+k3}")
# print(f"Ceq = {1/((1/c1)+(1/c2)+(1/c3))}")

maxBaseFreqHz = 0.21

baseFreqHz = min(np.sqrt(kRover/m), maxBaseFreqHz)
baseFreq = baseFreqHz/ (2 * math.pi)
baseFreq = 0

# print(f"Spring natural freq = {np.sqrt(k1/m)}")

def zcr(x, y):
    try:
        tt0 =  x[np.round(y, 5) <= 0.0][0]
        return tt0
    except:
        return Tmax

def wave(t, phase=0, bump=0, stop=Tmax, stop2=Tmax, baseFreq=baseFreq):
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


maxK = 2000
kList = np.arange(1,maxK,5)
kList = range(0,maxK,200)
FreqList = [0.21, 5, 10, 15, 21]
# FreqList = [21]

T = np.arange(0, Tmax, timestep)

motionFreqListList = []
accelListList = []
tt0ListList = []
surfaceAccelListList = []

for FreqIter in tqdm(FreqList):
    baseFreq = FreqIter * (2 * math.pi)
    waveW1 = wave(T, stop=int(stopLinear), stop2=int(stopToZero), baseFreq=baseFreq).tolist()
    waveW2 = wave(T, stop=int(stopLinear), stop2=int(stopToZero), baseFreq=baseFreq).tolist()
    waveW3 = wave(T, stop=int(stopLinear), stop2=int(stopToZero), baseFreq=baseFreq).tolist()

    accelList = []
    motionFreqList = []
    tt0List = []

    surfaceAccel = max((np.gradient(waveW1[1])/timestep))
    surfaceAccelListList.append(surfaceAccel)
    for kIter in tqdm(kList):
        k1 = kIter
        k2 = k1
        k3 = k1
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

        U = np.concatenate((waveW1, waveW2, waveW3), axis=0)

        ss = cm.StateSpace(A, B, C, D)
        # cm.damp(ss)

        TT, yout, xout = cm.forced_response(ss, T=T, X0=[0, 0, 0, 0], U=U, return_x=True)

        # dis = list(yout[0])
        vel1 = list(yout[1])
        accel1 = (np.gradient(vel1)/timestep)[500::]
        # angvelRads = list(np.round(yout[3], 15))
        # angAccelRads = (np.gradient(angvelRads)/timestep)[500::]
        # totalAccelleration = -(angAccelRads * (bodyWidth/2)) + accel1
        # print(max(accel1))

        accelList.append(max(accel1))
        motionFreqList.append(getFreq(accel1))
        

        # totAccelList.append(max(totalAccelleration))

        TT, yout, xout = cm.forced_response(ss, T=T, X0=[initialDisplacement, 0, initialRotation, 0], return_x=True)

        dis2 = list(yout[0])
        # vel2 = list(yout[1])
        # accel2 = (np.gradient(vel2)/timestep)[500::]

        # print()
        tt0List.append(zcr(T, np.array(dis2)))
        # plt.plot(T, dis2)
        # plt.plot(T[500:], accel1)
        # plt.show()

    accelListList.append(accelList)
    motionFreqListList.append(motionFreqList)
    tt0ListList.append(tt0List)

# print(motionFreqListList)
# print(len(motionFreqListList))
fig, axs = plt.subplots(3,1, sharex=False)
# axs[0].set_xscale("log")
# axs[1].set_xscale("log")
resonantFreq = 49.326

for index, frequency in enumerate(FreqList):
    axs[0].plot(kList, accelListList[index], label=f"CoM accel {frequency} Hz")
axs[2].plot(FreqList, [x[0] for x in motionFreqListList], label=f"CoM frequency")
axs[2].plot(FreqList, [resonantFreq for x in FreqList], label=f"CoM frequency")
# axs[0].plot(kList, [surfaceAccel for x in kList], label=f"Surface accel {21} Hz")
# axs[0].plot(kList, totAccelList, label="Max acceleration", color="red")

axs[1].plot(kList, tt0List)

axs[0].set_title(f"Acceleration")
axs[1].set_title(f"Time to rest from {maxAmplitude} displacement")

axs[1].set_xlabel("K N/m")
axs[1].set_ylabel("Time to rest [s]")
axs[0].set_ylabel("Acceleration [m/s^2]")

axs[0].grid()
axs[1].grid()

axs[0].legend()
# axs[1].legend()
fig.suptitle(f"K = 0 to {maxK}, freq = {round(baseFreq, 3)} rad/s, {baseFreqHz} Hz")
plt.show()
