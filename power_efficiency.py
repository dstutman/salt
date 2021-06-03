from matplotlib import pyplot as plt
import numpy as np
from math import radians,cos
import scipy as sp

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

    def num_cells(self, T):

        return self.P_req/self.cell_power(T)

    def cell_power(self, T): #temp in kelvin
        Vmp_new = self.Vmp + (self.DVmpdt*(T-(273.15+25)))
        Imp_new = self.Imp + (self.DImpdt * (T - (273.15 + 25)))

        return Vmp_new*Imp_new*cos(i)

    def rad_Q(self, T):
        stef_boltz = 5.67*10**(-8)
        Total_A = self.num_cells(T) * self.area

        return self.emiss*stef_boltz*T**4*Total_A*2

    def power_ab(self,T, i):
        Total_A = self.num_cells(T)*self.area

        return self.absor*self.solar*Total_A* cos(i)
    
    def solar_incidence(self,Js):
        self.solar = Js

    def power_req(self,P_req):
        self.P_req = P_req

    def Q_dif(self,T):
        Qab = self.power_ab(T, i)
        Qe = cell.rad_Q(T) +  self.P_req

        return abs(Qab-Qe)
        
        
P_req = 4045.26 # W, check if this is updated power required
i = radians(10)  #  incidence angle
Jsmin = 1321  # max solar incidence W/m^2
Jsavg =  1367  # average solar incidence W/m^2
Jsmax = 1412  # min solar incidence W/m^2

Js = Jsmax
cell = solar_cell()

Qdif = 10**6
## Min Q difference


for Js in [Jsmin, Jsavg, Jsmax]:
    Qdif = 10 ** 6
    print("Solar incidence = ", Js)
    cell.solar_incidence(Js)
    cell.power_req(P_req)

    for s in range(500000):
        T = 100 + s/1000
        Qab = cell.power_ab(T, i)
        Qe = cell.rad_Q(T) +  P_req

        if abs(Qab-Qe) < Qdif:
            Qdif = abs(Qab-Qe)
            Ncell = cell.num_cells(T)
            Teq = T

    print("Error heat = ",Qdif)
    print("Number cells = ",Ncell)
    print("Temperature equilibrium = ",Teq)
    print()
