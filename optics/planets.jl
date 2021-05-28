# Copyright Daniel Avishai Stutman 2021. All rights reserved.

using CSV
using DataFrames
using LinearAlgebra

# Configuration start
dataset = "PS_2021.05.20_08.13.20.csv"
prelim_filter = false
pt_auth = 90 # Pointing authority, degrees
# Configuration end

# Configuration protection
if pt_auth > 90
    println("Clamping pointing authority, was $pt_auth")
    pt_auth = 90
end
# Configuration protection end

# Start of program
println("Starting in $(pwd())")

# Load the dataset
println("Loading $dataset...")
ds = DataFrame(CSV.File("Datasets/$dataset", comment="#", select=[:soltype, :pl_controv_flag, :pl_orbsmax, :pl_rade, :pl_masse, :st_rad, :st_mass, :st_lum, :ra, :dec, :sy_dist]))

if prelim_filter
    # Filter out planets missing necessary fields, or that are unconfirmed or controversial
    #dropmissing!(ds, disallowmissing=true)
    filter!(row -> row[:soltype] == "Published Confirmed", ds)
    filter!(row -> row[:pl_controv_flag] == 0, ds)
end

# Set the default visibility modifier
@enum Visibility begin
    VISIBLE
    OUT_OF_VIEW
end
ds[:, :visibility] .= VISIBLE

# Convert right ascention and declination to radians
ds[:, :ra] = deg2rad.(ds[:, :ra])
ds[:, :dec] = deg2rad.(ds[:, :dec])

Rz(α) = [cos(α) sin(α) 0; -sin(α) cos(α) 0; 0 0 1]
Ry(δ) = [cos(δ) 0 -sin(δ); 0 1 0; sin(δ) 0 cos(δ)]

R(α, δ) = Ry(-δ) * Rz(α)

# Calculate the moon pointing vector
α0 = deg2rad(269.9949)
δ0 = deg2rad(66.5392)
p_moon = -(inv(R(α0, δ0)) * [1; 0; 0])

ds[:, :p_vect] = map((ra, dec) -> inv(R(ra, dec)) * [1; 0; 0], ds[:, :ra], ds[:, :dec])
ds[:, :sep_angle] = map(v -> acos(dot(v, p_moon)), ds[:, :p_vect])
ds[ds[:, :sep_angle] .> deg2rad(pt_auth), :visibility] .= OUT_OF_VIEW

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
scatter!(oof[1, :], oof[2, :], oof[3, :], color=:black, mopacity = 0.1, label="Out of FOR")

# Visible objects
vis = hcat(ds[ds[:, :visibility] .== VISIBLE, :p_vect]...)
scatter!(vis[1, :], vis[2, :], vis[3, :], color=:blue, label="Visible")


