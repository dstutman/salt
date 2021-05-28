# Copyright Daniel Avishai Stutman 2021. All rights reserved.

using CSV
using DataFrames
using LinearAlgebra

# Configuration start
dataset = "PS_2021.05.20_08.13.20.csv"
prelim_filter = true
pt_auth = 50 # Pointing authority, degrees
# Configuration end

# Configuration protection
if pt_auth > 90
    println("Clamping pointing authority, was $pt_auth")
    pt_auth = 90
end
# Configuration protection end

# SNR relationships
function null_depth(st_angular_dia, lambda=10E-6, baseline=1E3, phase_var=0, frac_intensity_var=0)
    st_leakage = pi^2 / 4 * (st_angular_dia/(lambda/baseline))^2
    return phase_var^2 + st_leakage + frac_intensity_var^2
end

function local_flux(bd_teq, bd_rad, sy_dist)
    boltzmann = 5.67E-8
    surface_flux = boltzmann * bd_teq^4
    local_flux = surface_flux * bd_rad^2 / sy_dist^2
    return local_flux
end
# SNR relationships end

# Start of program
println("Starting in $(pwd())")

# Load the dataset
println("Loading $dataset...")
ds = DataFrame(CSV.File("Datasets/$dataset", comment="#", select=[:soltype, :pl_controv_flag, :ra, :dec, :sy_dist, :pl_name, :pl_rade, :pl_eqt, :st_teff, :st_rad]))

if prelim_filter
    # Filter out planets missing necessary fields, or that are unconfirmed or controversial
    dropmissing!(ds, disallowmissing=true)
    filter!(row -> row[:soltype] == "Published Confirmed", ds)
    filter!(row -> row[:pl_controv_flag] == 0, ds)
end

# Set the default visibility modifier
@enum Visibility begin
    VISIBLE
    OUT_OF_VIEW
    OUT_OF_RANGE
end
ds[:, :visibility] .= VISIBLE

# Unit conversions
# Deg to rad
ds[:, :ra] = deg2rad.(ds[:, :ra])
ds[:, :dec] = deg2rad.(ds[:, :dec])
# Parsec to m
ds[:, :sy_dist] .*= 3.086E16
# Solar radii to m
ds[:, :st_rad] .*= 696340E3 
# Earth radii to m
ds[:, :pl_rade] .*= 6371E3

Rz(a) = [cos(a) sin(a) 0; -sin(a) cos(a) 0; 0 0 1]
Ry(d) = [cos(d) 0 -sin(d); 0 1 0; sin(d) 0 cos(d)]

R(α, δ) = Ry(-δ) * Rz(α)

# Calculate the moon pointing vector
a0 = deg2rad(269.9949)
d0 = deg2rad(66.5392)
p_moon = -(inv(R(a0, d0)) * [1; 0; 0])

# FOR filtering
ds[:, :p_vect] = map((ra, dec) -> inv(R(ra, dec)) * [1; 0; 0], ds[:, :ra], ds[:, :dec])
ds[:, :sep_angle] = map(v -> acos(dot(v, p_moon)), ds[:, :p_vect])
ds[ds[:, :sep_angle] .> deg2rad(pt_auth), :visibility] .= OUT_OF_VIEW

# Distance marking
# ds[(ds[:, :visibility] .== VISIBLE) .& (ds[:, :sy_dist] .> 50), :visibility] .= OUT_OF_RANGE

# SNR calculation
# Stellar noise
ds[:, :st_noise] = map(local_flux, ds[:, :st_teff], ds[:, :st_rad], ds[:, :sy_dist])
ds[:, :st_noise] .*= map((st_rad, sy_dist) -> null_depth(st_rad/sy_dist), ds[:, :st_rad], ds[:, :sy_dist])
# Planetary signal
ds[:, :pl_sig] = map(local_flux, ds[:, :pl_eqt], ds[:, :pl_rade], ds[:, :sy_dist])
ds[:, :pl_sig] .*= 0.5 # Getting planet signal modulation costs 1/2 signal



using Plots
plotly()

# ICRS origin
scatter([0], [0], [0], color=:orange, label="Solar Baricenter")

# Earth North Pole (sanity check)
scatter!([0], [0], [1], color=:yellow, label="Earth Celestial North")

# Moon south pole
scatter!([p_moon[1]], [p_moon[2]], [p_moon[3]], color=:red, label="Moon Celestial South")

# Out of FOR objects
oof = hcat(ds[ds[:, :visibility] .== OUT_OF_VIEW, :p_vect]...)
scatter!(oof[1, :], oof[2, :], oof[3, :], color=:black, mopacity = 0.2, label="Out of FOR")

# Out of range objects
# oor = hcat(ds[ds[:, :visibility] .== OUT_OF_RANGE, :p_vect]...)
# scatter!(oor[1, :], oor[2, :], oor[3, :], color=:blue, mopacity = 0.2, label="Out of range")

# Visible objects
vis = hcat(ds[ds[:, :visibility] .== VISIBLE, :p_vect]...)
scatter!(vis[1, :], vis[2, :], vis[3, :], color=:green, label="Visible")

title!("SALT Visiblity Study")
