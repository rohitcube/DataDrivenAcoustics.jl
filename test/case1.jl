using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using Flux
using Statistics
using Random
using AcousticsToolbox
using Plots # Optional: For debugging visualization


# ============================================================================
# TEST HELPER FUNCTIONS
# ============================================================================

"""
    validate_against_bellhop(model, pm_truth, tx, r_range, z_range;
                             frequency, soundspeed, max_pressure)

Validate a trained model against Bellhop ground truth on a grid.

# Arguments
- `model`: Trained PlaneWaveCurvModel
- `pm_truth`: Bellhop propagation model (ground truth)
- `tx`: AcousticSource
- `r_range`: Range values for validation grid
- `z_range`: Depth values for validation grid (negative = below surface)
- `frequency`: Frequency in Hz
- `soundspeed`: Sound speed in m/s
- `max_pressure`: Max pressure used for normalization

# Returns
- `rms_error`: RMS error in dB
"""
function validate_against_bellhop(model, pm_truth, tx,
                                  r_range, z_range;
                                  frequency, soundspeed,
                                  max_pressure)

    errors_db = Float64[]
    k = 2π * frequency / soundspeed

    for r in r_range
        for z in z_range
            # 1. Ground Truth (Bellhop) - in Pascals
            rx = AcousticReceiver(r, 0.0, z)
            rays_true = arrivals(pm_truth, tx, rx)
            p_true = isempty(rays_true) ? 0.0im : sum(ray.phasor for ray in rays_true)

            # 2. Model Prediction (normalized 0-1)
            coord = reshape([r, 0.0, z], 3, 1)
            p_pred_normalized = calculate_field(model, coord, k)[1]

            # 3. Denormalize: MULTIPLY by max_pressure to get Pascals
            p_pred = p_pred_normalized * max_pressure

            # 4. Calculate dB error (NO ALPHA - pure accuracy)
            db_true = 20 * log10(abs(p_true) + 1e-12)
            db_pred = 20 * log10(abs(p_pred) + 1e-12)

            push!(errors_db, (db_true - db_pred)^2)
        end
    end

    rms_error = sqrt(mean(errors_db))

    return rms_error
end


"""
    generate_zigzag_data(pm_truth, tx, r_range, z_range, n_points, n_profiles)

Generate training data in a zigzag pattern using Bellhop.

# Arguments
- `pm_truth`: Bellhop propagation model
- `tx`: AcousticSource
- `r_range`: Tuple (r_start, r_end) for range extent
- `z_range`: Tuple (z_min, z_max) for depth extent (negative = below surface)
- `n_points`: Total number of measurement points
- `n_profiles`: Number of zigzag legs

# Returns
- `train_r`: Vector of range coordinates
- `train_z`: Vector of depth coordinates
- `train_p`: Vector of complex pressure measurements
"""
function generate_zigzag_data(pm_truth, tx, r_range, z_range, n_points, n_profiles)

    r_start, r_end = r_range
    z_min, z_max = z_range

    r_leg_dist = (r_end - r_start) / n_profiles
    points_per_leg = div(n_points, n_profiles)

    train_r = Float64[]
    train_z = Float64[]
    train_p = ComplexF64[]

    println("   Starting ray tracing for training data...")

    # Validation: Test a single point first
    println("   → Testing Bellhop with first measurement point...")
    test_r, test_z = r_start, z_min
    rx_test = AcousticReceiver(test_r, 0.0, test_z)
    rays_test = arrivals(pm_truth, tx, rx_test)
    println("   → Bellhop returned $(length(rays_test)) rays for (r=$test_r, z=$test_z)")

    if isempty(rays_test)
        @warn "Bellhop returns 0 rays! Cannot generate training data."
        @warn "This usually means source/receiver geometry is invalid."
        error("Training data generation failed - Bellhop cannot compute ray paths")
    end
    println("   ✓ Bellhop validation passed")

    # Generate zigzag pattern
    for i in 1:n_profiles
        leg_r_start = r_start + (i-1) * r_leg_dist
        leg_r_end = leg_r_start + r_leg_dist

        # Alternate direction: Odd=shallow→deep, Even=deep→shallow
        leg_z_start, leg_z_end = isodd(i) ? (z_min, z_max) : (z_max, z_min)

        for j in 1:points_per_leg
            alpha = (j-1) / (points_per_leg - 1)
            r_curr = leg_r_start + alpha * (leg_r_end - leg_r_start)
            z_curr = leg_z_start + alpha * (leg_z_end - leg_z_start)

            push!(train_r, r_curr)
            push!(train_z, z_curr)

            # Get pressure from Bellhop
            rx = AcousticReceiver(r_curr, 0.0, z_curr)
            rays = arrivals(pm_truth, tx, rx)
            p_complex = isempty(rays) ? 0.0im : sum(ray.phasor for ray in rays)

            push!(train_p, p_complex)
        end
    end

    # Data quality checks
    n_zeros = count(abs.(train_p) .< 1e-15)
    n_nans = count(isnan.(train_p))
    println("   Generated $(length(train_p)) measurements.")
    println("   Data diagnostics:")
    println("     - Zero measurements: $n_zeros / $(length(train_p))")
    println("     - NaN measurements: $n_nans")
    println("     - R range: $(extrema(train_r))")
    println("     - Z range: $(extrema(train_z))")
    println("     - Pressure magnitude range: $(extrema(abs.(train_p)))")

    if n_nans > 0
        error("Training data contains NaN values!")
    end

    return train_r, train_z, train_p
end

# ============================================================================
# END TEST HELPER FUNCTIONS
# ============================================================================

@testset "Case 1: Range-Dependent Bathymetry (Bellhop)" begin

    println("\n" * "="^60)
    println("STARTING CASE 1 VALIDATION (With Normalization & Smart Init)")
    println("="^60)

    println("1. Setup Environment")
    # 1. Setup Environment
    f = 10_000.0
    c = 1541.0
    # Note: Using negative Z for depth per your convention
    env_truth = UnderwaterEnvironment(soundspeed=c, seabed=SandyClay, bathymetry=100.0)
    tx = AcousticSource(0.0, 0.0, -5.0, f)
    println("   Environment: Sound Speed = $(env_truth.soundspeed) m/s, Bathymetry = $(env_truth.bathymetry) m,
     Seabed = $(env_truth.seabed)")

    pm_truth = Bellhop(env_truth; nbeams=2000, min_angle=-10°, max_angle=10°)

    # 2. Generate Training Data
    println("2. Generating Zig-Zag Training Data...")
    train_r, train_z, train_p = generate_zigzag_data(
        pm_truth, tx, (1000.0, 1050.0), (-5.0, -30.0), 1000, 9
    )

    train_locs = stack_coordinates(train_r, train_z)

    # --- FIX 1: MAX NORMALIZATION (Critical for Gradient Stability) ---
    # We divide by the maximum pressure so targets are exactly 0.0 to 1.0.
    # This prevents the "Exploding Gradient" vs "Staying Quiet" conflict.
    max_p = maximum(abs.(train_p))
    train_meas_norm = reshape(train_p ./ max_p, 1, :)

    println("   [Data] Max Pressure in dataset: $max_p Pa")
    println("   [Data] Normalized Target Range: $(extrema(abs.(train_meas_norm)))")

    # 3. Train Model
    println("3. Training RBNN...")
    # Pass the NORMALIZED data to the environment
    env_dd = BasicDataDrivenUnderwaterEnvironment(
        train_locs, train_meas_norm;
        soundspeed=c, frequency=f, waterdepth=30.0
    )

    model = PlaneWaveCurvModel(env_dd, 60)

    # --- FIX 2: FIT WITH SMART ARGS ---
    loss_history = fit!(model, train_locs, train_meas_norm;
                        init_angles=:auto,
                        source_depth=-5.0,
                        max_epochs=5000,
                        learning_rate=0.01, # Slightly higher initial rate to fight L1
                        alpha=5e-3,         # <--- TRY THIS (Sparsity Penalty)
                        verbose=true,
                        log_interval=500)

    # 4. Validate
    println("4. Validating on Dense Grid...")
    val_r = range(1000.0, 1050.0, length=50)
    val_z = range(-5.0, -29.0, length=50)

    rms_error = validate_against_bellhop(
        model, pm_truth, tx,
        val_r, val_z,
        frequency=f, soundspeed=c,
        max_pressure=max_p  # Pass max_p directly
    )
    # 5. Results
    println("-"^40)
    println("   FINAL RESULTS")
    println("   RMS Error: $(round(rms_error, digits=2)) dB")
    println("-"^40)

    @test rms_error < 4.5
end
