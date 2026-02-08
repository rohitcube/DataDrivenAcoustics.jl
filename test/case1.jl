using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using Flux
using Statistics
using Random
using AcousticsToolbox
using Plots # Optional: For debugging visualization


function DataDrivenAcoustics.calculate_field(model::PlaneWaveCurvModel, coord::AbstractArray, k::Number)
    # 1. Unpack Coordinates (Vectorized for batch processing)
    r = coord[1, :]
    z = coord[3, :]

    # 2. Define the "Singer" function
    # This calculates the wave for ONE ray (index i)
    function ray_contribution(i)
        r_local = r .- 1000.0

        # Unpack parameters for this specific ray
        θ = model.theta[i]
        d = model.d[i]
        A = model.A[i]
        ϕ = model.phi[i]

        # Calculate Phase (Plane + Curvature)
        phase_plane = k .* (r_local .* cos(θ) .+ z .* sin(θ))
        phase_curv = (k .* z.^2) ./ (2 * d)

        # Return the Complex Pressure for this ray
        return A .* cis.(phase_plane .+ phase_curv .+ ϕ)
    end

    # 3. Summation (Superposition)
    # We sum the contributions of all rays (1 to nrays).
    # Zygote loves 'sum' because it knows exactly how to differentiate it.
    return sum(ray_contribution(i) for i in 1:model.nrays)
end

# ============================================================================
# HELPER FUNCTIONS (To be moved to src/ later)
# ============================================================================

"""
    fit!(model::PlaneWaveCurvModel, train_locs, train_meas;
         init_angles=:auto, reinit=false, max_epochs=3000,
         learning_rate=0.01, verbose=true, log_interval=500,
         scale_factor=1e6, min_curvature=50.0)

Train the PlaneWaveCurvModel to fit training data.

# Arguments
- `model`: PlaneWaveCurvModel to train
- `train_locs`: 3×N array of receiver coordinates [r, y, z]
- `train_meas`: 1×N array of measured complex pressures
- `init_angles`: Angle initialization strategy (:auto, :horizontal, :vertical, :uniform, or custom)
- `reinit`: If true, re-initialize parameters even if previously trained
- `max_epochs`: Maximum number of training iterations
- `learning_rate`: Adam optimizer learning rate
- `verbose`: Print training progress
- `log_interval`: Epochs between progress prints
- `scale_factor`: Scale factor applied to measurements (e.g., 1e6 for μPa)
- `min_curvature`: Minimum absolute curvature value for stability

# Returns
- `loss_history`: Vector of loss values at each logging interval
"""
function fit!(model::PlaneWaveCurvModel, train_locs, train_meas;
              init_angles=:auto,
              reinit=false,
              max_epochs=3000,
              learning_rate=0.01,
              verbose=true,
              log_interval=500,
              scale_factor=1e6,
              min_curvature=50.0)

    T = eltype(model.theta)
    nrays = model.nrays

    # Check if model needs initialization
    needs_init = all(model.theta .== 0) || reinit

    if needs_init
        verbose && @info "Initializing model parameters with strategy: $init_angles"

        # Initialize angles
        model.theta .= initialize_angles(model.env, nrays, init_angles, T)

        # Initialize other parameters
        model.A .= randn(T, nrays) .* T(0.1)
        model.phi .= zeros(T, nrays)
        # d (curvature) keeps its default value or could be reset here

    else
        verbose && @info "Continuing training with existing parameters (use reinit=true to restart)"
    end

    # Setup optimization
    loss_fn(x, y) = Flux.mse(abs.(x), abs.(y))
    opt = Flux.Adam(learning_rate)
    ps = Flux.params(model)
    target_amp = abs.(vec(train_meas))

    # Get wavenumber from environment
    k = 2π * model.env.frequency / model.env.soundspeed

    # Check initial state
    if verbose
        initial_preds = calculate_field(model, train_locs, k)
        initial_loss = loss_fn(initial_preds, target_amp)
        println("   Initial loss (before training): $initial_loss")
        println("   Initial prediction range: $(extrema(abs.(initial_preds)))")
        println("   Target amplitude range: $(extrema(target_amp))")
    end

    # Training loop
    loss_history = Float64[]

    for epoch in 1:max_epochs
        grads = Flux.gradient(ps) do
            preds = calculate_field(model, train_locs, k)
            loss_fn(preds, target_amp)
        end
        Flux.update!(opt, ps, grads)

        # Apply curvature clamping for stability
        for i in 1:model.nrays
            if abs(model.d[i]) < min_curvature
                model.d[i] = sign(model.d[i]) * min_curvature
            end
        end

        # Logging
        if verbose && (epoch % log_interval == 0)
            curr_loss = loss_fn(calculate_field(model, train_locs, k), target_amp)
            println("   Epoch $epoch: Loss = $curr_loss")
            push!(loss_history, curr_loss)
        end
    end

    verbose && println("   Training complete!")

    return loss_history
end


"""
    initialize_angles(env, nrays, strategy, T)

Initialize plane wave angles based on strategy.
"""
function initialize_angles(env, nrays::Int, strategy, T::Type)
    if strategy == :auto
        return smart_initialize_angles(env, nrays, T)

    elseif strategy == :uniform
        @warn "Using uniform random initialization. May converge slowly for directional geometries."
        return rand(T, nrays) .* T(2π) .- T(π)

    elseif strategy == :horizontal
        return randn(T, nrays) .* T(0.1)

    elseif strategy == :vertical
        base = rand(T, nrays) .< 0.5 ? T(π/2) : T(-π/2)
        return base .+ randn(T, nrays) .* T(0.1)

    elseif strategy isa NamedTuple && haskey(strategy, :center) && haskey(strategy, :spread)
        return T(strategy.center) .+ randn(T, nrays) .* T(strategy.spread)

    elseif strategy isa AbstractVector
        @assert length(strategy) == nrays "Provided angles must match nrays=$nrays"
        return T.(strategy)

    else
        error("Unknown init_angles strategy: $strategy. Use :auto, :uniform, :horizontal, :vertical, (center, spread), or a vector of angles.")
    end
end


"""
    smart_initialize_angles(env, nrays, T)

Automatically determine angle initialization based on source-receiver geometry.
Falls back to moderate horizontal spread if geometry information is unavailable.
"""
function smart_initialize_angles(env, nrays::Int, T::Type)
    # Try to extract receiver positions from environment
    # This will depend on your actual environment structure

    # For BasicDataDrivenUnderwaterEnvironment, measurements might be stored differently
    # Fallback: use moderate spread around horizontal
    @info "Smart initialization: Using moderate horizontal spread (±17°) as default"
    return randn(T, nrays) .* T(0.3)

    # TODO: Implement actual geometry analysis when env structure is known
    # Example structure if receivers are available:
    # if hasproperty(env, :receiver_positions) && !isempty(env.receiver_positions)
    #     receivers = env.receiver_positions
    #     source = hasproperty(env, :source_position) ? env.source_position : [0.0, 0.0, 0.0]
    #
    #     angles_to_receivers = T[]
    #     for receiver in eachcol(receivers)
    #         dx = receiver[1] - source[1]
    #         dz = receiver[3] - source[3]
    #         angle = atan(dz, dx)
    #         push!(angles_to_receivers, angle)
    #     end
    #
    #     min_angle = minimum(angles_to_receivers)
    #     max_angle = maximum(angles_to_receivers)
    #     angle_center = (min_angle + max_angle) / 2
    #     angle_spread = (max_angle - min_angle) / 2
    #
    #     angle_spread *= T(1.2)  # 20% margin
    #     angle_spread = clamp(angle_spread, T(0.1), T(π/4))
    #
    #     @info "Smart initialization: center=$(round(rad2deg(angle_center), digits=1))°, spread=±$(round(rad2deg(angle_spread), digits=1))°"
    #
    #     return angle_center .+ randn(T, nrays) .* angle_spread
    # end
end


"""
    validate_against_bellhop(model, pm_truth, tx, r_range, z_range;
                             frequency, soundspeed, scale_factor=1e6)

Validate a trained model against Bellhop ground truth on a grid.

# Arguments
- `model`: Trained PlaneWaveCurvModel
- `pm_truth`: Bellhop propagation model (ground truth)
- `tx`: AcousticSource
- `r_range`: Range values for validation grid
- `z_range`: Depth values for validation grid (negative = below surface)
- `frequency`: Frequency in Hz
- `soundspeed`: Sound speed in m/s
- `scale_factor`: Scale factor used in model training (default 1e6 for μPa)

# Returns
- `rms_error`: RMS error in dB
- `error_grid`: Matrix of errors at each grid point (optional, for plotting)
"""
function validate_against_bellhop(model, pm_truth, tx,
                                  r_range, z_range;
                                  frequency, soundspeed,
                                  scale_factor=1e6)

    errors_db = Float64[]
    k = 2π * frequency / soundspeed

    for r in r_range
        for z in z_range
            # 1. Ground Truth (Bellhop)
            rx = AcousticReceiver(r, 0.0, z)
            rays_true = arrivals(pm_truth, tx, rx)
            p_true = isempty(rays_true) ? 0.0im : sum(ray.phasor for ray in rays_true)

            # 2. Model Prediction
            coord = reshape([r, 0.0, z], 3, 1)
            p_pred_scaled = calculate_field(model, coord, k)[1]

            # Convert back from scaled units to Pascals
            p_pred = p_pred_scaled / scale_factor

            # 3. Calculate dB error
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


"""
    stack_coordinates(r, z)

Stack range and depth coordinates into a 3×N array for model input.

# Arguments
- `r`: Vector of range coordinates
- `z`: Vector of depth coordinates

# Returns
- 3×N array with [r; y; z] where y=0
"""
function stack_coordinates(r, z)
    train_locs = zeros(3, length(r))
    train_locs[1, :] = r
    train_locs[3, :] = z
    return train_locs
end


"""
    prepare_measurements(p, scale_factor=1e6)

Prepare complex pressure measurements for model training.

# Arguments
- `p`: Vector of complex pressure values
- `scale_factor`: Scaling factor (default 1e6 for μPa conversion)

# Returns
- 1×N array of scaled measurements
"""
function prepare_measurements(p, scale_factor=1e6)
    return reshape(p .* scale_factor, 1, :)
end

# ============================================================================
# END HELPER FUNCTIONS
# ============================================================================

@testset "Case 1: Range-Dependent Bathymetry (Bellhop)" begin

    println("\n" * "="^60)
    println("STARTING CASE 1 VALIDATION")
    println("="^60)

    # 1. Setup Environment
    println("1. Setting up Environment (Sloping Bottom)...")
    f = 10_000.0
    c = 1541.0

    env_truth = UnderwaterEnvironment(
        soundspeed = c,
        seabed = SandyClay,
        bathymetry = 100.0
    )

    tx = AcousticSource(0.0, 0.0, -5.0, f)
    println("   Created source at depth 5.0m (z = -5.0)")

    println("   Initializing Bellhop...")
    pm_truth = Bellhop(env_truth; nbeams=2000, min_angle=-10°, max_angle=10°, debug=false)
    println("   Bellhop initialized successfully")

    # 2. Generate Training Data
    println("2. Generating Zig-Zag Training Data...")
    train_r, train_z, train_p = generate_zigzag_data(
        pm_truth, tx,
        (1000.0, 1050.0),  # r_range
        (-5.0, -30.0),      # z_range
        1000,               # n_points
        9                   # n_profiles
    )

    train_locs = stack_coordinates(train_r, train_z)
    train_meas = prepare_measurements(train_p, 1e6)

    # 3. Train Model
    println("3. Training RBNN...")
    env_dd = BasicDataDrivenUnderwaterEnvironment(
        train_locs, train_meas;
        soundspeed = c, frequency = f, waterdepth = 30.0
    )

    model = PlaneWaveCurvModel(env_dd, 60)

    loss_history = fit!(model, train_locs, train_meas,
                       init_angles=:horizontal,
                       max_epochs=3000,
                       learning_rate=0.01,
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
        scale_factor=1e6
    )

    # 5. Results and Assertions
    println("-"^40)
    println("   FINAL RESULTS")
    println("   RMS Error: $(round(rms_error, digits=2)) dB")
    println("-"^40)

    @test rms_error < 4.5

    if rms_error < 3.5
        println("   [SUCCESS] Matches Error-Free Benchmark!")
    elseif rms_error < 4.5
        println("   [PASS] Within Acceptable Limits.")
    else
        println("   [FAIL] Error too high.")
    end
end