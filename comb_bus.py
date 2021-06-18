from math import radians, cos
import scipy.optimize as sp

# constants
sig = 5.67*10**-8  # stefan boltzmann constant

# lunar regolith properties
moon_flux = 2.75 # W/m^2
Eps_lun = 0.95   # emissivity
k_lun = 7.4*10**-4  # conductivity  monte carlo this
T_lun = 35  # minimal lunar surface temperature in Kelvin
t = 0.005  # thickness of layer with regolith in meters
t_bottom = 5  # thickness of layer in the bottom

ems_louver_closed = 0.1  #
ems_louver_open = 0.8
ems_mat = 0.07

# Box dimensions
length = 1.78  # meters  monte carlo
width = 0.54   # meters monte carlo
height = 0.21  # meters monte carlo

Th = 258 + 32  # temperature of the electronics in Kelvin
Qdis = 80.9  # dissipated heat in Watts


# box areas
A_louvers = height*(2*length+width)
A_connects = height*width
Atop= (length+2*t)*(width+2*t)
Asides = (height+t)*2*(length+2*t+width+2*t)
Abottom = length*width

Acon = length*width #+ height*2*(length + width)

Arad = Atop #+Asides


def conduction(T):
    return k_lun*Acon/t*(Th-T)


def radiation(T):
    return Eps_lun*Arad*sig*T**4


def conduction_bottom():
    return (Th-T_lun)*k_lun*Abottom/t_bottom


def func(T):
    return radiation(T)-conduction(T)


Teq = sp.root(func, [1600])


def heat_difference(closed):
    if closed:
        return Eps_lun*sig*Arad*Teq.x[0]**4+conduction_bottom()-Qdis + ems_louver_closed*A_louvers*sig*Th**4 + ems_mat*A_connects*sig*Th**4 #- ems_louver_closed*A_louvers*moon_flux -ems_mat * A_connects*moon_flux
    else:
        return Eps_lun * sig * Arad * Teq.x[0] ** 4 + conduction_bottom() - Qdis + ems_louver_open * A_louvers * sig * Th ** 4 + ems_mat * A_connects * sig * Th ** 4 #- ems_louver_open*A_louvers*moon_flux -ems_mat * A_connects*moon_flux


print("Error heat = ", Teq.qtf[0])
print("Temperature outer layer= ", Teq.x[0])
print("Loss heat radiation= ", Eps_lun*sig*Arad*Teq.x[0]**4)
print("Conduction bottom = ", conduction_bottom())
print()
print("Louvers closed")
print("Required heating(positive) or cooling(negative) = ", heat_difference(True))
print()
print("Louvers open")
print("Required heating(positive) or cooling(negative) = ", heat_difference(False))
print()