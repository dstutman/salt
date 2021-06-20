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
initialDisplacement = 0.1

debug = False
plot = True
animate = True
save = False

maxAmplitude = 0.02  # [m]
# baseFreq = 0.1

m = 1750
J = 125  # = 56.55cos(11deg) + 70.3cos(11deg) (ixx+iyy)

l1 = 0.9
l2 = 0
l3 = -l1

kRover = 1000000000

# k2 must be less than 40*k1 or 40*k2
k1 = 5000
k2 = 5000
k3 = 5000

c1 = m * 0.95 * 3
c2 = m * 0.95 * 3
c3 = m * 0.95 * 3

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


maxK = 150000
minK = 10000
intervalK = 2000
kList = np.arange(1,maxK,5)
kList = range(minK, maxK, intervalK)
# FreqList = [0.21,1 ,2, 5, 10, 21]
FreqList = [0.21, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1, 5, 10, 21]
FreqList = [0.21,0.35, 0.5, 0.8, 1, 10, 21]
FreqList = [21]

T = np.arange(0, Tmax, timestep)

motionFreqListList = []
accelListList = []
dispListList = []
velListList = []
tt0ListList = []
surfaceAccelListList = []
i=0
for FreqIter in tqdm(FreqList):
    baseFreq = FreqIter * (2 * math.pi)
    waveW1 = wave(T, stop=int(stopLinear), stop2=int(stopToZero), baseFreq=baseFreq).tolist()
    waveW2 = wave(T, stop=int(stopLinear), stop2=int(stopToZero), baseFreq=baseFreq).tolist()
    waveW3 = wave(T, stop=int(stopLinear), stop2=int(stopToZero), baseFreq=baseFreq).tolist()

    accelList = []
    dispList = []
    velList = []
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

        dis1 = list(yout[0])
        vel1 = list(yout[1])
        accel1 = (np.gradient(vel1)/timestep)[500::]
        # angvelRads = list(np.round(yout[3], 15))
        # angAccelRads = (np.gradient(angvelRads)/timestep)[500::]
        # totalAccelleration = -(angAccelRads * (bodyWidth/2)) + accel1
        # print(max(accel1))

        accelList.append(max(accel1))
        dispList.append(max(dis1))
        velList.append(max(vel1))
        motionFreqList.append(getFreq(accel1))
        

        # totAccelList.append(max(totalAccelleration))
        i
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
    dispListList.append(dispList)
    velListList.append(velList)
    motionFreqListList.append(motionFreqList)
    tt0ListList.append(tt0List)

# print(motionFreqListList)
# print(len(motionFreqListList))
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
# axs[0].set_xscale("log")
# axs[1].set_xscale("log")
resonantFreq = 49.326
maxAccelMirrors = 9*9.81


maxRecoveryTime = 10
for i, u in enumerate(tt0List):
    if u < maxRecoveryTime:
        iRecovery = i
        break

# print(tt0ListList)
# print(tt0List)

for index, frequency in enumerate(FreqList):
    axs[0].plot([x/2 for x in kList], accelListList[index], label=f"CoM accel {frequency} Hz")
    axs[1].plot([x/2 for x in kList], velListList[index], label=f"CoM vel {frequency} Hz")
    axs[2].plot([x/2 for x in kList], dispListList[index], label=f"CoM {frequency} Hz")
# axs[2].plot(FreqList, [x[0] for x in motionFreqListList], label=f"CoM frequency")
# axs[2].plot(FreqList, [resonantFreq for x in FreqList], label=f"CoM frequency")
axs[0].plot(kList, [surfaceAccel for x in kList], label=f"Surface accel {21} Hz")
# axs[0].plot(kList, [maxAccelMirrors for x in kList], label=f"maxAccel mirrors {maxAccelMirrors}m/s^2")
# axs[0].plot([kList[i] for x in range(2)], [x for x in (0, maxAccelMirrors)], label=f"Max recovery time ({maxRecoveryTime}s)")
# axs[0].plot(kList, totAccelList, label="Max acceleration", color="red")

# axs[3].plot([kList[i] for x in range(2)], [x for x in (0, 40)], label=f"Max recovery time ({maxRecoveryTime}s)")
axs[3].plot([x/2 for x in kList], tt0List)

# axs[0].set_title(f"Acceleration")
# axs[1].set_title(f"Time to rest from {maxAmplitude} displacement")

axs[3].set_xlabel("K [N/m]")
axs[3].set_ylabel("Time to rest [s]")
axs[0].set_ylabel("Acceleration [m/s^2]")
axs[1].set_ylabel("Velocity [m/s]")
axs[2].set_ylabel("Displacement [m]")

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()

# axs[0].legend()
axs[2].legend(loc='lower right')
# axs[3].legend()
# fig.suptitle(f"K = {minK} to {maxK}, freq = {min(FreqList)} Hz to  {max(FreqList)} Hz")
plt.show()
