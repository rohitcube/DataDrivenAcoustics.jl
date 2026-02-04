using DataDrivenAcoustics
using UnderwaterAcoustics
using Serialization

println("Generating Pekeris Waveguide Ground Truth Data...")
println("=" ^ 70)

# ==========================================================================
# SETUP THE PHYSICS (Same as test)
# ==========================================================================
f = 5000.0
c_water = 1500.0
water_depth = 100.0

env_phys = UnderwaterEnvironment(
    soundspeed = c_water,
    bathymetry = water_depth,
    seabed = SandySilt
)

pm = PekerisRayTracer(env_phys)
tx = AcousticSource(0.0, 0.0, 50.0, f)

# ==========================================================================
# GENERATE TRAINING DATA
# ==========================================================================
println("Generating training data (168 points)...")

range_train = [100.0, 105.0]
depths_train = range(5.0, 95.0, length=84)
n_points = length(range_train) * length(depths_train)

train_locs = zeros(Float64, 3, n_points)
train_meas = zeros(ComplexF64, 1, n_points)

idx = 1
for r in range_train
    for z in depths_train
        global idx  # Declare idx as global for script-level scope
        rx = AcousticReceiver(r, 0.0, z)
        rays = arrivals(pm, tx, rx)
        p_complex = sum(a.phasor for a in rays)

        train_locs[1, idx] = r
        train_locs[2, idx] = 0.0
        train_locs[3, idx] = z
        train_meas[1, idx] = p_complex
        idx += 1
    end
end

println("  ✓ Training data generated: $n_points sensors")

# ==========================================================================
# GENERATE VALIDATION DATA
# ==========================================================================
println("Generating validation data...")

test_r = 102.5
test_z = 50.0

rx_test = AcousticReceiver(test_r, 0.0, test_z)
rays_test = arrivals(pm, tx, rx_test)
p_true = sum(a.phasor for a in rays_test)

println("  ✓ Validation data generated")

# ==========================================================================
# SAVE TO DISK
# ==========================================================================
fixtures_dir = joinpath(@__DIR__, "fixtures")
mkpath(fixtures_dir)

fixture_path = joinpath(fixtures_dir, "pekeris_case2.dat")

data = Dict(
    "train_locs" => train_locs,
    "train_meas" => train_meas,
    "test_r" => test_r,
    "test_z" => test_z,
    "p_true" => p_true,
    "f" => f,
    "c_water" => c_water,
    "water_depth" => water_depth,
    "tx" => tx
)

open(fixture_path, "w") do io
    serialize(io, data)
end

println("\n" * "=" ^ 70)
println("✓ Fixtures saved to: $fixture_path")
println("  File size: $(round(filesize(fixture_path) / 1024, digits=2)) KB")
println("\nYou can now run the tests with pre-loaded data!")
