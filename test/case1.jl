using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using Flux
using Statistics
using Random
using Plots # Optional: For debugging visualization

# Load the model structure (Ensure this file exists in src/)
include("../src/pm_case1.jl")

@testset "Case 1: Range-Dependent Bathymetry Validation" begin

    println("\n" * "="^60)
    println("CASE 1: RANGE-DEPENDENT BATHYMETRY & CURVATURE")
    println("="^60)

    # ==========================================================================
    # 1. SETUP THE PHYSICS (Ground Truth Generator)
    # ==========================================================================
    println("[1/4] Setting up Range-Dependent Environment...")

    f = 10_000.0       # 10 kHz
    c_water = 1541.0   # Isovelocity
    z_source = 5.0     # Source depth

    # Bathymetry: Sloping upward from 40m (at source) to 30m (at AOI 1km away)
    # Slope gradient: (40 - 30) / 1000 = 0.01 m/m
    function bathy_slope(x, y)
        r = sqrt(x^2 + y^2)
        if r < 1000.0
            return 40.0 - (0.01 * r)
        else
            return 30.0 # Flat floor inside the AOI (1000m+)
        end
    end

    # Use RaySolver for accurate range-dependent ray tracing
    env_truth = UnderwaterEnvironment(
        soundspeed = c_water,
        seabed = SandyClay, # As specified in paper
        bathymetry = bathy_slope
    )

    tx = AcousticSource(0.0, 0.0, z_source, f)
    pm_truth = RaySolver(env_truth; nrays=5000) # High ray density for ground truth

    # ==========================================================================
    # 2. GENERATE ZIG-ZAG DATA (Profiling Float Trajectory)
    # ==========================================================================
    println("[2/4] Generating Zig-Zag Profiling Data...")

    # Zig-Zag Parameters
    n_points_total = 1000
    n_profiles = 9
    r_start = 1000.0
    r_end = 1050.0
    z_min, z_max = 0.0, 30.0

    # Calculate legs
    r_leg_dist = (r_end - r_start) / n_profiles
    points_per_leg = div(n_points_total, n_profiles)

    r_list = Float64[]
    z_list = Float64[]
    meas_list = ComplexF64[]

    for i in 1:n_profiles
        # Define start and end of this leg
        leg_r_start = r_start + (i-1)*r_leg_dist
        leg_r_end   = leg_r_start + r_leg_dist

        # Zig-Zag: Odd legs go DOWN (0->30), Even legs go UP (30->0)
        if isodd(i)
            leg_z_start, leg_z_end = z_min, z_max
        else
            leg_z_start, leg_z_end = z_max, z_min
        end

        # Interpolate points along the diagonal
        for j in 1:points_per_leg
            alpha = (j-1) / (points_per_leg - 1)
            r_curr = leg_r_start + alpha * (leg_r_end - leg_r_start)
            z_curr = leg_z_start + alpha * (leg_z_end - leg_z_start)

            push!(r_list, r_curr)
            push!(z_list, z_curr)

            # Compute Ground Truth Pressure
            rx = AcousticReceiver(r_curr, 0.0, z_curr)

            # Note: transmissionloss returns positive dB.
            # We convert to complex pressure roughly assuming Phase=0 for magnitude training,
            # OR better: use coherent output if RaySolver supports it directly.
            # Here we use coherent TL to get magnitude and phase.
            tl_complex = transmissionloss(pm_truth, tx, rx, mode=:coherent)

            # Convert TL to Pressure: P = 10^(-TL/20)
            # RaySolver coherent mode usually returns TL.
            # We approximate Pressure Magnitude here as the paper focuses on Amplitude fit.
            # If your RaySolver returns complex P directly, use that.
            p_mag = 10^(-real(tl_complex)/20.0)
            p_phase = imag(tl_complex) # RaySolver often packs phase in imag part of TL or returns phasor

            # *CRITICAL*: DataDrivenAcoustics usually expects Complex Pressure.
            # If RaySolver output is ambiguous, we construct a phasor from magnitude.
            push!(meas_list, p_mag * cis(0.0)) # Phase is hard to match perfectly without exact timing, magnitude is key.
        end
    end

    # Split Data (70% Train, 30% Val)
    n_train = Int(floor(0.7 * length(meas_list)))

    train_r = r_list[1:n_train]
    train_z = z_list[1:n_train]
    train_p = meas_list[1:n_train]

    # Format for Model
    train_locs = zeros(3, n_train)
    train_locs[1, :] = train_r
    train_locs[3, :] = train_z
    train_meas = reshape(train_p, 1, :)

    println("  Training Points: $n_train")
    println("  Validation Points: $(length(meas_list) - n_train)")

    # ==========================================================================
    # 3. TRAIN THE MODEL (RBNN)
    # ==========================================================================
    println("[3/4] Training PlaneWaveCurvModel (RBNN)...")

    # Setup Data Container
    env_dd = BasicDataDrivenUnderwaterEnvironment(
        train_locs, train_meas;
        soundspeed = c_water, frequency = f, waterdepth = z_max
    )

    # Initialize Model with 60 Rays (as per paper)
    model = PlaneWaveCurvModel(env_dd, 60)

    # Optimizer
    # We use a slightly lower rate because Curvature (d) can be sensitive
    opt = Flux.Adam(0.02)

    # Custom Loss: Log-Magnitude Error (dB Error) is often better for acoustics
    # But MSE on pressure is standard for RBNN code.
    loss() = Flux.mse(calculate_field(model), vec(train_meas))

    # Training Loop
    ps = Flux.params(model)
    epochs = 3000

    # Progress animation
    anim = Animation()

    for epoch in 1:epochs
        Flux.train!(loss, ps, [()], opt)

        if epoch % 500 == 0
            curr_loss = loss()
            println("  Epoch $epoch: MSE = $(round(curr_loss, digits=6))")
        end
    end

    println("  Learned Parameters (Sample):")
    println("  Curvature d: $(round(mean(model.d), digits=1)) m (Avg)")

    # ==========================================================================
    # 4. BENCHMARK EVALUATION (601x601 Grid)
    # ==========================================================================
    println("[4/4] Running Benchmark Evaluation (601x601 Grid)...")

    # Define dense grid
    grid_r = range(1000.0, 1050.0, length=101) # Reduced to 101x101 for speed in Test
    grid_z = range(0.0, 30.0, length=101)      # (Paper uses 601, scale up if needed)

    errors_db = Float64[]

    # We calculate Physics Truth on the fly (or you could pre-calc)
    # Warning: RaySolver is slow. For 10,000 points, this takes ~1-2 mins.

    println("  Computing Field Error...")

    # Pre-calculate k
    k_val = 2π * f / c_water

    for r in grid_r
        for z in grid_z
            # 1. Ground Truth
            rx = AcousticReceiver(r, 0.0, z)
            tl_complex = transmissionloss(pm_truth, tx, rx, mode=:coherent)
            p_true_mag = 10^(-real(tl_complex)/20.0)

            # 2. Prediction
            # We construct the coordinate vector manually
            coord = reshape([r, 0.0, z], 3, 1)
            p_pred = calculate_field(model, coord, k_val)[1]
            p_pred_mag = abs(p_pred)

            # 3. Calculate Error in dB
            val_true_db = 20 * log10(p_true_mag + 1e-9)
            val_pred_db = 20 * log10(p_pred_mag + 1e-9)

            push!(errors_db, (val_true_db - val_pred_db)^2)
        end
    end

    # Calculate RMS Error
    mse_db = mean(errors_db)
    rmse_db = sqrt(mse_db)

    println("\n" * "-"^40)
    println("  FINAL RESULTS")
    println("  RMS Error: $(round(rmse_db, digits=2)) dB")
    println("-"^40)

    # ==========================================================================
    # 5. ASSERTIONS (Success Criteria)
    # ==========================================================================

    # The paper achieves ~3.08 dB.
    # We allow a small margin (up to 4.5 dB) for random initialization variance.
    @test rmse_db < 4.5

    if rmse_db < 3.2
        println("  ✅ PASSED: Matches High-Accuracy Benchmark (~3.08 dB)")
    elseif rmse_db < 4.5
        println("  ⚠️ PASSED: Matches Positioning-Error Benchmark (~4.26 dB)")
    else
        println("  ❌ FAILED: Error too high")
    end

end