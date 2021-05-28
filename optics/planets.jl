# Copyright Daniel Avishai Stutman 2021. All rights reserved.

using CSV
using DataFrames
using LinearAlgebra

# Configuration start
dataset = "PS_2021.05.20_03.16.32.csv"
# Configuration end

# Start of program
println("Starting in $(pwd())")

# Load the dataset
println("Loading $dataset...")
ds = DataFrame(CSV.File("Datasets/$dataset", comment="#", select=[:soltype, :pl_controv_flag, :pl_orbsmax, :pl_rade, :pl_masse, :st_rad, :st_mass, :st_lum, :ra, :dec, :sy_dist]))

# Filter out planets missing necessary fields, or that are unconfirmed or controversial
#dropmissing!(ds, disallowmissing=true)
filter!(row -> row[:soltype] == "Published Confirmed", ds)
filter!(row -> row[:pl_controv_flag] == 0, ds)

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

ds[:, :p_vect] = map((ra, dec) -> tuple((inv(R(ra, dec)) * [1; 0; 0])...), ds[:, :ra], ds[:, :dec])
ds[:, :sep_angle] = map(v -> acos(dot(v, p_moon)), ds[:, :p_vect])

using Plots
xyz = reinterpret(reshape, Int, ds[ds[:, :sep_angle] .< 30, :p_vect])
scatter(xyz...)