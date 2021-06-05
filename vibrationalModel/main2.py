import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as cm
import math
import scipy as sc
from matplotlib.animation import FuncAnimation

s = np.sign

Tmax = 6
stopLinear = Tmax
stopToZero = Tmax
randomFreq=0
timestep = 0.001 # do not change

debug = True
plot = True
animate = True
save = False

maxAmplitude = 0.02  # [m]
# baseFreq = 0.1

m = 1500/2
J = 125 # = 56.55cos(11deg) + 70.3cos(11deg) (ixx+iyy)

l1 = 1.2
l2 = 0.2
l3 = -1.2

kRover = 1000000000

# k2 must be less than 40*k1 or 40*k2 
k1 = 200
k2 = k1
k3 = k1

c1 = 0.9*m
c2 = 0.9*m
c3 = 0.9*m

s1 = s(l1)
s2 = s(l2)
s3 = 1

maxBaseFreqHz = 21

baseFreqHz = min(np.sqrt(kRover/m), maxBaseFreqHz)
baseFreq = baseFreqHz/ (2 * math.pi)

print(f"Spring natural freq = {np.sqrt(k1/m)}")

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

if debug:
    print("Start Matrix unit test")
    print("A thetadotdot", A[3])
    print("B thetadotdot", B[3])
    print(f"s1 = {s1}, s2 = {s2}, s3 = {s3},")
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

initialRotation = 0

waveW1 = wave(T, phase1, bump=bump1, stop=int(stopLinear), stop2=int(stopToZero)).tolist()
waveW2 = wave(T, phase2, bump=bump2, stop=int(stopLinear), stop2=int(stopToZero)).tolist()
waveW3 = wave(T, phase3, bump=bump3, stop=int(stopLinear), stop2=int(stopToZero)).tolist()

U = np.concatenate((waveW1, waveW2, waveW3), axis=0)

ss = cm.StateSpace(A, B, C, D)
cm.damp(ss)

TT, yout, xout = cm.forced_response(ss, T=T, X0=[0, 0, initialRotation, 0], U=U, return_x=True)

dis = list(yout[0])
vel = list(yout[1])
angdis = list(np.round(np.degrees(yout[2]), 15))
angvel = list(np.round(np.degrees(yout[3]), 15))

### Plotting
fig, axs = plt.subplots(2, 2)
accelFig, accelAx = plt.subplots(1, 1)

fig.suptitle(f"m={round(m, 3)},k1={round(k1, 3)}, c1={round(c1, 3)}, c1/m={round(c1/m,3)}, freq={round(baseFreq, 3)}[rad/s]/{round(baseFreqHz, 3)}[Hz]")
accelFig.suptitle("Acceleration")
axs[0, 0].plot(T, waveW1[0], label="moonquake displacement w1", linestyle="--", color="orange")
axs[0, 0].plot(T, dis, label="displacement", color="tab:blue")

axs[0, 1].plot(T, waveW1[1], label="moonquake velocity w1", linestyle="--", color="orange")
axs[0, 1].plot(T, vel, label="velocity", color="tab:blue")
# axs[0, 1].plot(T[100::], (np.gradient(dis)/timestep)[100::], label="test vello") # verifying the acceleration graph

axs[1, 0].plot(T, angdis, label="ang displacement", color="tab:blue")

axs[1, 1].plot(T[100::], (np.gradient(angvel)/timestep)[100::], label="ang accelleration", color="g")
axs[1, 1].plot(T, angvel, label="ang velocity", color="tab:blue")

accelAx.plot(T[100::], (np.gradient(waveW1[1])/timestep)[100::], label="surface accelleration", color="orange")
accelAx.plot(T[100::], (np.gradient(vel)/timestep)[100::], label="accelleration", color="g")

axs[0, 0].legend(loc="lower right")
axs[0, 1].legend(loc="lower right")
axs[1, 0].legend(loc="lower right")
axs[1, 1].legend(loc="lower right")
accelAx.legend(loc="lower right")

axs[0, 0].grid()
axs[0, 1].grid()
axs[1, 0].grid()
axs[1, 1].grid()
accelAx.grid()

fig.tight_layout()
accelFig.tight_layout()

### Animations
if animate:
    anm, anfig = plt.subplots(1, 1, figsize=(6, 6))

    llen = 1.2 # length of lines
    llen2 = -1.2
    llline1, = anfig.plot( [0, llen*math.cos(math.radians(0))], [0, llen*math.sin(math.radians(0))], "k-", lw=2)
    llline2, = anfig.plot( [0, llen2*math.cos(math.radians(0))], [0, llen2*math.sin(math.radians(0))], "k-", lw=2)

    cgDot, = anfig.plot([0], [0], 'ro')
    cgOrigin, = anfig.plot([0], [0], 'g2')

    l1, l2, l3 = -l1, -l2, -l3 #original coordinate system is mirrored in up axis

    w1, = anfig.plot([l1], [-0.5], 'ro')
    w2, = anfig.plot([l2], [-0.5], 'go')
    w3, = anfig.plot([l3], [-0.5], 'bo')
    ow1, = anfig.plot([l1], [-0.5], 'b2')
    ow2, = anfig.plot([l2], [-0.5], 'r2')
    ow3, = anfig.plot([l3], [-0.5], 'g2')

    anfig.set_xlim([-1.5,1.5])
    anfig.set_ylim([-1.5,1.5])

    def animate(i):
        cgDot.set_data(0, dis[int(i*1000)])

        w1.set_data(l1, waveW1[0][int(i*1000)]-0.5 + bump1)
        w2.set_data(l2, waveW2[0][int(i*1000)]-0.5 + bump2)
        w3.set_data(l3, waveW3[0][int(i*1000)]-0.5 + bump3)

        llline1.set_data([0, llen*math.cos(math.radians(-1*angdis[int(i*0)]))],
                        [dis[int(i*1000)], llen * math.sin(math.radians(-1*angdis[int(i*1000)]))+dis[int(i*1000)]])

        llline2.set_data([0, llen2*math.cos(math.radians(-1*angdis[int(i*0)]))],
                        [dis[int(i*1000)], llen2 * math.sin(math.radians(-1*angdis[int(i*1000)]))+dis[int(i*1000)]])

        anfig.set_title(f"{round(i,3)}s")

        return cgDot, w1, w2, w3, llline1, llline2

    anim = FuncAnimation(anm, animate, frames=T[::10], interval=1, blit=True, repeat=True)

    if save:
        anim.save(f"animations/021.mp4", fps=120, extra_args=['-vcodec', 'libx264'], dpi=200)
if plot:
    plt.show()
