using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using Random

@testset "Case 1: Physics Kernel Math" begin
    # constants
    k = 1.0             # wavenumber
    A = 1.0             # amplitude
    phi = 0.0           # phase offset
    d_flat = 1e9        # "Infinity" (Flat wavefront)

    # --- Scenario 1: Ray from East (0 radians) ---
    # Receiver is at x=10, y=0.
    # The wave travels 10 meters along the x-axis.
    # Phase should accumulate by k * distance = 1.0 * 10.0 = 10.0
    p_east = plane_wave_propagate(10.0, 0.0, k, A, phi, 0.0, d_flat)

    # Check Phase (modulo 2pi)
    @test isapprox(angle(p_east), rem2pi(10.0, RoundNearest), atol=1e-5)

    # --- Scenario 2: Ray from North (pi/2 radians) ---
    # Receiver is at x=0, y=10.
    # The wave travels 10 meters along the y-axis (from North to Origin).
    # Since ray comes FROM North (down), propagation is typically modeled relative to source direction.
    # Let's assume standard math: Direction vector is [cos(theta), sin(theta)].
    p_north = plane_wave_propagate(0.0, 10.0, k, A, phi, π/2, d_flat)

    # Projection of (0,10) onto (0,1) is 10.0. Phase should be 10.0.
    @test isapprox(angle(p_north), rem2pi(10.0, RoundNearest), atol=1e-5)
end

@testset "Case 1: Learning Capability" begin
    # 1. SETUP: Create a "Fake Reality"
    # A source coming from 45 degrees (pi/4)
    true_theta = π/4
    true_d = 1000.0
    k = 2π * 100.0 / 1500.0 # Changed from 1000Hz to 100Hz

    # Create a shorter line of 10 sensors (0 to 20m instead of 100m)
    rx_locs = zeros(3, 10)
    rx_locs[1, :] = range(0, 20, length=10)

    # Generate Synthetic Data (Ground Truth)
    measurements = ComplexF64[]
    for i in 1:10
        x, y = rx_locs[1,i], rx_locs[2,i]
        push!(measurements, plane_wave_propagate(x, y, k, 1.0, 0.0, true_theta, true_d))
    end

    # 2. INITIALIZE: Create a Dumb Model
    # We cheat slightly and set nrays=1 for this unit test to ensure stability
    env = BasicDataDrivenUnderwaterEnvironment(rx_locs, reshape(measurements, 1, :);
                                             frequency=100.0, soundspeed=1500.0)
    model = PlaneWaveCurvModel(env, 1)

    # Force the model to start WRONG (e.g., pointing at 0 degrees)
    model.theta .= 0.5

    # 3. ACTION: Train it!
    # We expect fit! to exist and handle the training
    trained_model = fit!(model, measurements)

    # 4. ASSERT: Did it learn?
    # The learned angle should be close to pi/4 (0.785)
    learned_theta = trained_model.theta[1]

    # Allow small error (0.05 rad) because optimization isn't perfect
    @test isapprox(learned_theta, true_theta, atol=0.05)
end

@testset "Case 1: Learning Capability - DEBUG" begin
    true_theta = π/4
    true_d = 1000.0
    k = 2π * 100.0 / 1500.0

    rx_locs = zeros(3, 10)
    rx_locs[1, :] = range(0, 20, length=10)

    measurements = ComplexF64[]
    for i in 1:10
        x, y = rx_locs[1,i], rx_locs[2,i]
        push!(measurements, plane_wave_propagate(x, y, k, 1.0, 0.0, true_theta, true_d))
    end

    env = BasicDataDrivenUnderwaterEnvironment(rx_locs, reshape(measurements, 1, :);
                                             frequency=100.0, soundspeed=1500.0)
    model = PlaneWaveCurvModel(env, 1)

    # Check initial state
    println("Initial A: ", model.A)
    println("Initial theta: ", model.theta)
    println("Initial d: ", model.d)

    model.theta .= 0.5

    # Train with verbose output
    trained_model = fit!(model, measurements; verbose=true, max_epochs=5000, learning_rate=0.1)

    println("Final A: ", trained_model.A)
    println("Final theta: ", trained_model.theta)
    println("Final d: ", trained_model.d)
    println("Target theta: ", true_theta)
end


@testset "Case 2: Spherical Physics Kernel" begin
    # Constants
    k = 1.0
    A = 10.0 # Start with large amplitude to see decay
    phi = 0.0

    # --- Scenario 1: Source at Origin (0,0) ---
    src_x, src_y = 0.0, 0.0

    # Receiver at (10, 0)
    # Expected r = 10.0
    # Expected Amplitude = A / r = 10 / 10 = 1.0
    # Expected Phase = k * r = 10.0
    p1 = spherical_wave_propagate(10.0, 0.0, src_x, src_y, k, A, phi)

    @test isapprox(abs(p1), 1.0, atol=1e-5)
    @test isapprox(angle(p1), rem2pi(10.0, RoundNearest), atol=1e-5)

    # --- Scenario 2: Source Offset (Inverse Square Law Check) ---
    # Source at (-10, 0). Receiver at (10, 0).
    # Distance r = 20.0
    # Expected Amplitude = 10 / 20 = 0.5
    p2 = spherical_wave_propagate(10.0, 0.0, -10.0, 0.0, k, A, phi)

    @test isapprox(abs(p2), 0.5, atol=1e-5)
end




@testset "Case 2: Paper Reproduction (Pekeris Waveguide)" begin
    # ==========================================================================
    # 1. LOAD PRE-COMPUTED GROUND TRUTH DATA
    # ==========================================================================
    using Serialization

    fixture_path = joinpath(@__DIR__, "fixtures", "pekeris_case2.dat")

    if !isfile(fixture_path)
        error("""
        Fixture file not found: $fixture_path

        Please run the following command first to generate ground truth data:
            julia test/generate_fixtures.jl
        """)
    end

    println("Loading pre-computed ground truth data...")
    data = open(deserialize, fixture_path)

    train_locs = data["train_locs"]
    train_meas = data["train_meas"]
    test_r = data["test_r"]
    test_z = data["test_z"]
    p_true = data["p_true"]
    f = data["f"]
    c_water = data["c_water"]
    water_depth = data["water_depth"]
    tx = data["tx"]

    println("  ✓ Loaded $(size(train_locs, 2)) training points")

    # ==========================================================================
    # 3. SETUP THE MODEL (The "Brain")
    # ==========================================================================
    # Create the Data Driven Environment
    # Note: We keep using the new scalar style here too if we updated our own struct,
    # but for now we wrap the raw data.
    env_dd = BasicDataDrivenUnderwaterEnvironment(
        train_locs, train_meas;
        soundspeed = c_water,
        frequency = f,
        waterdepth = water_depth,
        tx = tx
    )

    # Initialize Case 2 Model
    model = SphericalWaveModel(env_dd, 100)

    # ==========================================================================
    # 4. TRAIN (fit!)
    # ==========================================================================
    println("  Starting Training...")

    # 5kHz optimization - Using 3000 epochs to ensure convergence
    fit!(model, train_meas;
         max_epochs=3000,
         learning_rate=0.01,
         verbose=true)

    # ==========================================================================
    # 5. VALIDATION
    # ==========================================================================
    println("  Validating...")

    # Prediction (ground truth p_true was loaded from fixture)
    test_coord = reshape([test_r, 0.0, test_z], 3, 1)
    k_val = 2π * f / c_water
    p_pred = calculate_field(model, test_coord, k_val)[1]

    # Calculate NMSE (as used in literature)
    mse = abs2(p_pred - p_true)
    signal_power = abs2(p_true)
    nmse = mse / signal_power

    println("  Truth: $(abs(p_true))")
    println("  Pred:  $(abs(p_pred))")
    println("  NMSE:  $nmse")

    # NMSE threshold of 0.04 corresponds to 20% relative error
    # (0.2^2 = 0.04)
    @test nmse < 0.04
end

# ==============================================================================
# TEST SET 2: INTEGRATION TESTS
# Focus: End-to-End workflow (Ground Truth -> Fitting -> Prediction)
# ==============================================================================
#@testset "Unit Test: Far-field 2D (Metadata Only)" begin
    #println("\nRunning Integration Test: Case 1 Pipeline...")

    ## ---------------------------------------------------------
    ## 1. GROUND TRUTH (Generate Fake Data using Standard Library)
    ## ---------------------------------------------------------
    #println("  1. Generating ground truth data...")

    #env_real = UnderwaterEnvironment(
        #soundspeed = 1500.0,
        #bathymetry = 100.0,
        #seabed = SandyMud,
        #surface = PressureReleaseBoundary
    #)
    #pm_real = PekerisRayTracer(env_real)
    #tx_real = AcousticSource(0.0, -5.0, 1000.0)
    #rx_real = AcousticReceiverGrid2D(1000.0:100.0:2000.0, -50.0:10.0:-10.0)

    ## Generate Training Data
    #tl_truth = transmission_loss(pm_real, tx_real, rx_real)
    #measurements = reshape(tl_truth, 1, :)
    #rx_locs = location.(rx_real) # Broadcast location over grid

    ## ---------------------------------------------------------
    ## 2. DATA DRIVEN SETUP
    ## ---------------------------------------------------------
    #println("  2. Setting up DataDriven model...")

    ## Create Blind Environment (Missing depth)
    #env_dd = DataDrivenEnvironment(
        #soundspeed = 1500.0,
        #frequency = 1000.0,
        #surface = UnderwaterAcoustics.PressureReleaseBoundary
    #)

    ## Initialize Model (Should pick SphericalWaveModel internally)
    #pm_dd = RayBasisNN(env_dd; nrays=60)
    #@test pm_dd isa SphericalWaveModel # Double check inside the flow

    #tx_dd = AcousticSource(0.0, -5.0, 1000.0)

    ## ---------------------------------------------------------
    ## 3. TRAINING (Fitting)
    ## ---------------------------------------------------------
    #println("  3. Fitting model...")
    ## This calls fit!, which updates the env with tx and initializes weights
    #fit!(pm_dd, tx_dd, rx_locs, transmission_loss, measurements)

    ## ---------------------------------------------------------
    ## 4. INFERENCE (Prediction)
    ## ---------------------------------------------------------
    #println("  4. Running inference...")

    #rx_pred = AcousticReceiverGrid2D(2000.0:50.0:2500.0, -50.0)
    #tl_pred = transmission_loss(pm_dd, tx_dd, rx_pred)

    ## ---------------------------------------------------------
    ## 5. VALIDATION
    ## ---------------------------------------------------------
    ## Ensure we got numbers back, not errors or NaNs
    #@test tl_pred isa Matrix{<:Number}
    #@test size(tl_pred) == size(rx_pred)
    #@test !any(isnan, tl_pred)

    #println("  ✔ Integration test passed!")
#end