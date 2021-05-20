using Markdown
using InteractiveUtils
using CSV
using DataFrames

# Configuration start
dataset = "PS_2021.05.20_03.16.32.csv"

# Configuration end

# Start of program
println("Starting in $(pwd())")

# Load the dataset
println("Loading $dataset...")
ds = DataFrame(CSV.File("Datasets/$dataset", comment="#", select=[:soltype, :pl_controv_flag, :pl_orbsmax, :pl_rade, :pl_masse, :pl_insol, :st_rad, :st_mass, :st_lum, :ra, :dec, :sy_dist]))

# Filter out planets missing necessary fields, or that are unconfirmed or controversial
dropmissing!(ds, disallowmissing=true)
filter!(row -> row[:soltype] == "Published Confirmed", ds)
filter!(row -> row[:pl_controv_flag] == 0, ds)