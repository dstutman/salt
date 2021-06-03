from matplotlib import pyplot as plt
import numpy as np
from math import radians,cos

class solar_cell():
    def __init__(self):
        # all the properties of the used solar cell
        self.emiss = 0.85
        self.absor = 0.91
        self.Vmp = 2.793  # V
        self.Imp = 0.4238  # A
        self.DVmpdt = -0.009  # V/K
        self.DImpdt = 0.00007  # A/K
        self.area = 30.18*10**-4  # m**2

    def num_cells(self, Req_power, T):

        return Req_power/self.cell_power(T)

    def cell_power(self, T): #temp in kelvin
        Vmp_new = self.Vmp + (self.DVmpdt*(T-(273.15+25)))
        Imp_new = self.Imp + (self.DImpdt * (T - (273.15 + 25)))

        return Vmp_new*Imp_new*cos(i)

    def rad_Q(self, T):
        stef_boltz = 5.67*10**(-8)
        Total_A = self.num_cells(P_req, T) * self.area

        return self.emiss*stef_boltz*T**4*Total_A*2

    def power_ab(self,P_req,T, Js, i):
        Total_A = self.num_cells(P_req, T)*self.area

        return self.absor*Js*Total_A* cos(i)

P_req = 728 # W, check if this is updated power required
i = radians(10)  #  incidence angle
Js = 1367  # Solar incidence W/m^2

cell = solar_cell()

step_max = 1  # K
Qdif = 10**6
## Min Q difference

for s in range(5000):
    T = 100 + s/10
    Qab = cell.power_ab(P_req, T, Js, i)
    rad = cell.rad_Q(T)
    Qe = cell.rad_Q(T) +  P_req

    if abs(Qab-Qe) < Qdif:
        Qdif = abs(Qab-Qe)
        Ncell = cell.num_cells(P_req, T)
        Teq = T

print(Qdif)
print(Ncell)
print(Teq)
