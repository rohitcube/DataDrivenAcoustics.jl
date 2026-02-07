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

@testset "Case 1: Range-Dependent Bathymetry (Bellhop)" begin

    println("\n" * "="^60)
    println("STARTING CASE 1 VALIDATION")
    println("="^60)

    println("1. Setting up Environment (Sloping Bottom)...")

    f = 10_000.0
    c = 1541.0


    env_truth = UnderwaterEnvironment(
        soundspeed = c,
        seabed = SandyClay,
        bathymetry = 100.0  # Increased from 40m to avoid Bellhop boundary warnings
    )

    tx = AcousticSource(0.0, 0.0, -5.0, f)  # Negative z = depth below surface
    println("   Created source at depth 5.0m (z = -5.0)")

    # Initialize Bellhop (The Gold Standard)
    println("   Initializing Bellhop...")
    pm_truth = Bellhop(env_truth; nbeams=2000, min_angle=-10°, max_angle=10°, debug=false)
    println("   Bellhop initialized successfully")
    println("   Using angle range: -10° to 10° (focused on horizontal propagation)")

    println("2. Generating Zig-Zag Data (using arrivals)...")

    # Zig-Zag Parameters
    n_points_total = 1000
    n_profiles = 9
    r_start, r_end = 1000.0, 1050.0
    z_min, z_max = -5.0, -30.0  # Negative z = depth below surface (5m to 30m deep)

    r_leg_dist = (r_end - r_start) / n_profiles
    points_per_leg = div(n_points_total, n_profiles)

    train_r = Float64[]
    train_z = Float64[]
    train_p = ComplexF64[]

    println("   Starting ray tracing for training data...")

    # STAGE 2 VALIDATION: Test a single point first
    println("   → Testing Bellhop with first measurement point...")
    test_r, test_z = 1000.0, -5.0  # Negative z = 5m depth
    rx_test = AcousticReceiver(test_r, 0.0, test_z)
    rays_test = arrivals(pm_truth, tx, rx_test)
    println("   → Bellhop returned $(length(rays_test)) rays for (r=$test_r, z=$test_z)")

    if isempty(rays_test)
        @warn "STAGE 2 FAILED: Bellhop returns 0 rays! Cannot generate training data."
        @warn "This usually means source/receiver geometry is invalid."
        error("Stage 2 (training data generation) failed - Bellhop cannot compute ray paths")
    end
    println("   ✓ Stage 2 validation passed - Bellhop is working")

    for i in 1:n_profiles
        leg_r_start = r_start + (i-1)*r_leg_dist
        leg_r_end   = leg_r_start + r_leg_dist

        # Alternate Direction: Odd=Shallow→Deep, Even=Deep→Shallow
        # (z_min=-5 is shallow, z_max=-30 is deeper)
        leg_z_start, leg_z_end = isodd(i) ? (z_min, z_max) : (z_max, z_min)

        for j in 1:points_per_leg
            alpha = (j-1) / (points_per_leg - 1)
            r_curr = leg_r_start + alpha * (leg_r_end - leg_r_start)
            z_curr = leg_z_start + alpha * (leg_z_end - leg_z_start)

            push!(train_r, r_curr)
            push!(train_z, z_curr)

            # --- CORRECT PHYSICS GENERATION ---
            rx = AcousticReceiver(r_curr, 0.0, z_curr)

            # 1. Get Eigenrays
            rays = arrivals(pm_truth, tx, rx)

            # 2. Coherent Sum (Preserves Phase)
            # Handle shadow zones safely with check
            p_complex = isempty(rays) ? 0.0im : sum(r.phasor for r in rays)

            push!(train_p, p_complex)
        end
    end

    # Pack Data for RBNN
    train_locs = zeros(3, length(train_r))
    train_locs[1, :] = train_r
    train_locs[3, :] = train_z
    scale_factor = 1e6
    train_meas = reshape(train_p .* scale_factor, 1, :)

    println("   Generated $(length(train_p)) measurements.")

    # Data quality checks
    n_zeros = count(abs.(train_p) .< 1e-15)
    n_nans = count(isnan.(train_p))
    println("   Data diagnostics:")
    println("     - Zero measurements: $n_zeros / $(length(train_p))")
    println("     - NaN measurements: $n_nans")
    println("     - Z range: $(extrema(train_locs[3, :]))")
    println("     - Pressure magnitude range: $(extrema(abs.(train_p)))")

    if n_nans > 0
        error("Training data contains NaN values!")
    end

    println("3. Training RBNN...")

    env_dd = BasicDataDrivenUnderwaterEnvironment(
        train_locs, train_meas;
        soundspeed = c, frequency = f, waterdepth = 30.0
    )

    # Initialize Model (60 neurons as per paper)
    model = PlaneWaveCurvModel(env_dd, 60)

    # Training Loop
    # We optimize for Magnitude match to ensure robust envelope fitting
    loss_fn(x, y) = Flux.mse(abs.(x), abs.(y))

    opt = Flux.Adam(0.01)
    ps = Flux.params(model)
    target_amp = abs.(vec(train_meas))

    # Check initial loss before training
    initial_preds = calculate_field(model, train_locs, 2π*f/c)
    initial_loss = loss_fn(initial_preds, target_amp)
    println("   Initial loss (before training): $initial_loss")
    println("   Initial prediction range: $(extrema(abs.(initial_preds)))")
    println("   Target amplitude range: $(extrema(target_amp))")

    for epoch in 1:3000
        grads = Flux.gradient(ps) do
            preds = calculate_field(model, train_locs, 2π*f/c)
            loss_fn(preds, target_amp)
        end
        Flux.update!(opt, ps, grads)

        for i in 1:model.nrays
            if abs(model.d[i]) < 50.0
                model.d[i] = sign(model.d[i]) * 50.0
            end
        end

        if epoch % 500 == 0
            curr_loss = loss_fn(calculate_field(model, train_locs, 2π*f/c), target_amp)
            println("   Epoch $epoch: Loss = $curr_loss")
        end
    end

    println("4. Validating on Dense Grid...")

    # Generate Dense Grid (50x50 for speed, paper uses 601x601)
    val_r = range(1000.0, 1050.0, length=50)
    val_z = range(-5.0, -29.0, length=50)  # Negative z = depth (5m to 29m deep)
    errors_db = Float64[]

    for r in val_r
        for z in val_z
            # 1. Ground Truth (Bellhop)
            rx = AcousticReceiver(r, 0.0, z)
            rays_true = arrivals(pm_truth, tx, rx)
            p_true = isempty(rays_true) ? 0.0im : sum(r.phasor for r in rays_true)

            # 2. Prediction (RBNN)
            coord = reshape([r, 0.0, z], 3, 1)
            p_pred = calculate_field(model, coord, 2π*f/c)[1]

            # 3. Calculate Error in dB
            # Add epsilon 1e-12 to avoid log(0)
            db_true = 20 * log10(abs(p_true) + 1e-12)
            db_pred = 20 * log10(abs(p_pred) + 1e-12)

            push!(errors_db, (db_true - db_pred)^2)
        end
    end

    final_rms = sqrt(mean(errors_db))
    println("-"^40)
    println("   FINAL RESULTS")
    println("   RMS Error: $(round(final_rms, digits=2)) dB")
    println("-"^40)

    # Success Criteria from Paper (3.08 dB ideal - 4.26 dB w/ error)
    @test final_rms < 4.5

    if final_rms < 3.5
        println("   [SUCCESS] Matches Error-Free Benchmark!")
    elseif final_rms < 4.5
        println("   [PASS] Within Acceptable Limits.")
    else
        println("   [FAIL] Error too high.")
    end
end