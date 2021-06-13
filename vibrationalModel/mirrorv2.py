import math

mirrorDiameter = 2
a = (mirrorDiameter/2)/5
# h = 10 / 1000
# h = 3 / 1000

# material -
# rho = 2210
# v = 0.17
# E = 67.6e9

rho = 3000
v = 0.21
E = 460e9

n = 0
s = 1

requiredNatFreq = 21*5

def alphaf(n, s):
    return (math.pi/2) * (n + 2*s)

def m(n):
    return 4 * n**2

def lmbda(n, s):
    mm = m(n)
    alpha = alphaf(n,s)

    # return alpha - ((mm+1) / (8*alpha)) - ((4 * (7*mm ** 2 + 22*mm + 11)) / (3 * (8 * alpha)**3))
    return alpha - ((mm+1) / (8*alpha)) - (4 * (7*mm**2+22*mm+11))/(3*(8*alpha)**3)


def D(E, h, v):
    return (E*h*h*h)/(12*(1-(v*v)))


def omega(lmda, a, rho, D):
    return (lmda**2)/(a**2 * math.sqrt(rho/D))

DD = D(E, h, v)
lmbda = lmbda(n, s)
print(f"Flexural rigidity = {DD}")
print(f"Lambda squared = {lmbda**2}")
print(f"Natural freq  = {omega(lmbda, a, rho, DD)}")
# print(f"Lambda squared = {lambdsqrd}")
