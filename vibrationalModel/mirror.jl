# feels weird without imports

mirrorDiameter = 1.5
a = (mirrorDiameter/2)/5
h = 32 / 1000

# material - 
ρ = 2210
v = 0.17
E = 67.6e9

n = 1


function α(n, s=0)
    return (pi/2)*(n+2*s)
end


function m(n)
    return 4*n^2
end


function lmbda(n, s=0)
    # return (pi/2) * (n + 2*s)
    mm = m(n)
    alpha = α(n,s)
    return alpha - ((mm + 1)/ (8*alpha)) - (4 * ( 7*mm^2 + 22*mm + 11) )/ ( 3*(8 *alpha)^3)
end


function omega(λ, a, ρ, D)
    return λ^2/(a^2 * sqrt(ρ/D))
end


function D(E,h,v)
    return E*h^3/(12*(1-v^2))
end

DD = D(E,h,v)

n = 1
println("18 segments, Mirror diameter: ", mirrorDiameter, "m")
println("r = ", a, "m E = ", E, "GPa v = ", v, " h = ", h, "m ρ = ", ρ, "kg/m^3")
println()
for s in 0:1:10  
    local λ = lmbda(n, s)
    local ω = omega(λ, a, ρ, DD)/ 2pi
    print("ω= ")
    print(round(ω, digits=3), "Hz")
    print(", n=")
    print(n)
    print(", s=")
    print(s)
    println(", f_",n,"/",s)
    """
    YES IM A 10X Programmer
    """
end