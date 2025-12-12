using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using Random

# ==============================================================================
# TEST SET 1: UNIT TESTS
# Focus: Testing logic in isolation (Does the Factory pick the right Model?)
# ==============================================================================
@testset "Unit Test: Model Selection Logic" begin
    println("Running Unit Test: Model Picker...")

    # Scenario A: Metadata Only (No depth provided)
    # Expected Result: Should pick 'SphericalWaveModel' (The Blank Slate / Plane Wave model)
    env_meta = DataDrivenEnvironment(
        soundspeed=1500.0,
        frequency=1000.0,
        waterdepth=missing # <--- The key trigger
    )

    # Call the picker function
    model_meta = RayBasisNN(env_meta)

    # ASSERTION: Verify the type of the returned object
    @test model_meta isa SphericalWaveModel
    println("  ✔ Correctly selected SphericalWaveModel for metadata-only environment.")
end

# ==============================================================================
# TEST SET 2: INTEGRATION TESTS
# Focus: End-to-End workflow (Ground Truth -> Fitting -> Prediction)
# ==============================================================================
@testset "Integration Test: Far-field 2D (Metadata Only)" begin
    println("\nRunning Integration Test: Case 1 Pipeline...")

    # ---------------------------------------------------------
    # 1. GROUND TRUTH (Generate Fake Data using Standard Library)
    # ---------------------------------------------------------
    println("  1. Generating ground truth data...")

    env_real = UnderwaterEnvironment(
        soundspeed = 1500.0,
        bathymetry = 100.0,
        seabed = SandyMud,
        surface = PressureReleaseBoundary
    )
    pm_real = PekerisRayTracer(env_real)
    tx_real = AcousticSource(0.0, -5.0, 1000.0)
    rx_real = AcousticReceiverGrid2D(1000.0:100.0:2000.0, -50.0:10.0:-10.0)

    # Generate Training Data
    tl_truth = transmission_loss(pm_real, tx_real, rx_real)
    measurements = reshape(tl_truth, 1, :)
    rx_locs = location.(rx_real) # Broadcast location over grid

    # ---------------------------------------------------------
    # 2. DATA DRIVEN SETUP
    # ---------------------------------------------------------
    println("  2. Setting up DataDriven model...")

    # Create Blind Environment (Missing depth)
    env_dd = DataDrivenEnvironment(
        soundspeed = 1500.0,
        frequency = 1000.0,
        surface = UnderwaterAcoustics.PressureReleaseBoundary
    )

    # Initialize Model (Should pick SphericalWaveModel internally)
    pm_dd = RayBasisNN(env_dd; nrays=60)
    @test pm_dd isa SphericalWaveModel # Double check inside the flow

    tx_dd = AcousticSource(0.0, -5.0, 1000.0)

    # ---------------------------------------------------------
    # 3. TRAINING (Fitting)
    # ---------------------------------------------------------
    println("  3. Fitting model...")
    # This calls fit!, which updates the env with tx and initializes weights
    fit!(pm_dd, tx_dd, rx_locs, transmission_loss, measurements)

    # ---------------------------------------------------------
    # 4. INFERENCE (Prediction)
    # ---------------------------------------------------------
    println("  4. Running inference...")

    rx_pred = AcousticReceiverGrid2D(2000.0:50.0:2500.0, -50.0)
    tl_pred = transmission_loss(pm_dd, tx_dd, rx_pred)

    # ---------------------------------------------------------
    # 5. VALIDATION
    # ---------------------------------------------------------
    # Ensure we got numbers back, not errors or NaNs
    @test tl_pred isa Matrix{<:Number}
    @test size(tl_pred) == size(rx_pred)
    @test !any(isnan, tl_pred)

    println("  ✔ Integration test passed!")
end