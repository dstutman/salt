using Plots

# Telescope parameters
fov = 42*4.8481368110954E-9
λ = 10e-6
bu = 500
bv = 400

# Calculate the angular period
Tu = λ/bu
Tv = λ/bv

# Arbitrary range of thetas for plotting
θu = collect((-fov/2):(Tu/100):(fov/2))
θv = collect((-fov/2):(Tu/100):(fov/2))

# Plot the responses (outer product of individual nulling)
r = cos.((1/Tv)*θv .- π) * transpose(cos.((1/Tu)*θu)) .+ 1

# Scale the response exponentially for easier viewing of the nulls
r = exp.(r.*2)

# Shift the zero response back to zero
r .-= 1

# Normalize the result
r ./= (maximum(r)-minimum(r))

# Plot the result
heatmap(θu, θv, r, title="Expected Interference Pattern bu=$bu bv=$bv", xlabel="\\theta_u [rad]", ylabel="\\theta_v [rad]", colorbar=:none, aspect_ratio=:equal)