import math

mirrorDiameter = 2
nSegments = 18
a = (mirrorDiameter/2)/5

a = a / (math.sqrt(3) / 2)

# a = 1030/2000
# a = 1350/2000
# a = 257.5/2000
area = math.sqrt(6.5) * a**2
print(f"Full mirror diameter = {2}, {nSegments} Segments")
print(f"Radius a={round(a, 4)}m, {round(a, 4)*1000}mm")
lmd2 = 8.892
lmd2 = 5.253
wn = 180

n = 0 
s = 1

lightweighted = 80
rhoOriginal = 3400
rho = rhoOriginal * ((100-lightweighted)/100)
# rho = 303
v = 0.21
E = 460e9

D = rho / (lmd2 / (wn * a**2)**2)

h = ((12 * D * (1-v**2))/E)**(1/3)

print(f"Original density = {rhoOriginal}, lightweighting = {lightweighted}, final density = {rho}")
# print(f"D = {D}")
# print(v)
# print(E)
print(f"Resonant freq max = {wn} Hz")
print(f"Resonant freq minimum thickness = {round(h,5)}m, {round(h,5)*1000} mm\n")
# print(f"lmd2 = {lmd2}, lmd = {math.sqrt(lmd2)}")

m = 1500
acc = 2.5*9.81
# acc = 10
sigmayCore = 470e6
sigmayMirror = 49.8e6

sigmayMax = min(sigmayCore, sigmayMirror)

# h2 = math.sqrt((3*m*acc)/sigmayMax)

h2 = math.sqrt( ((3*((m*acc)*(3+v)))/(8*sigmayMax)) )

ULEthicknessFactor = 0.103
rhoULE = 2210
arealDensity = 10

hfinal = max(h,h2)
hfinal = 31.5/1000
tf = hfinal*ULEthicknessFactor
print(tf)
D = (E*hfinal*hfinal*hfinal)/(12*(1-(v*v)))
D = (E*hfinal*hfinal*tf)/(2*(1-(v*v)))
# print(D)
wnfinal = (lmd2)/(a**2 * math.sqrt(rho/D))
massfinal = (rho * area * hfinal) + (rhoULE * area * hfinal * ULEthicknessFactor)
# massfinal = (arealDensity * area) + (rhoULE * area * hfinal * ULEthicknessFactor)
print(f"Max acceleration = {round(acc, 4)}")
print(f"Area = {round(area, 3)} m^2")
print(f"Final Thickness = {(round(hfinal,4))}m, {(round(hfinal,4))*1000} mm")
print(f"Natural frequency final = {wnfinal} Hz")
print(f"Mass final (w ULE) = {round(massfinal,5)} kg")
print(f"Total mass for {nSegments} Segments = {nSegments *round(massfinal,5)} kg")
g = 9.81
p = (g * massfinal) / area
delta = math.exp(1.267 * a/a) * (p/D) * a**4
f = (1.26/(2 * math.pi)) * (g/delta)**2
print(f)
