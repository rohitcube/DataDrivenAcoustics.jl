# ============================================================================
# Case 1: PlaneWaveCurvModel — field calculation, training, and utilities
# ============================================================================

function calculate_field(model::PlaneWaveCurvModel, coord::AbstractArray, k::Number)
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


"""
    initialize_angles(env, nrays, strategy, T; source_depth=0.0)

Initialize plane wave angles based on strategy.
"""
function initialize_angles(env, nrays::Int, strategy, T::Type; source_depth=0.0)
    if strategy == :auto || strategy == :smart_cone
        return smart_initialize_angles(env, nrays, T; source_depth=source_depth)

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
    smart_initialize_angles(env, nrays, T; source_depth=0.0)

Automatically determine angle initialization based on source-receiver geometry.
Falls back to moderate horizontal spread if geometry information is unavailable.
"""
function smart_initialize_angles(env, nrays::Int, T::Type; source_depth=0.0)
    # 1. Extract Receiver Locations (3xN matrix)
    if !hasproperty(env, :locations)
         @warn "Environment has no locations. Defaulting to horizontal."
         return randn(T, nrays) .* T(0.1)
    end

    r_vals = env.locations[1, :]
    z_vals = env.locations[3, :]

    # 2. Calculate Angle to every receiver: atan(dz, dr)
    direct_angles = atan.(z_vals .- source_depth, r_vals)

    # 3. Define the Cone
    min_ang, max_ang = minimum(direct_angles), maximum(direct_angles)
    center_ang = (min_ang + max_ang) / 2

    # Width: Cover the receivers + 15 degrees extra for surface/bottom bounces
    half_width = max((max_ang - min_ang)/2 * 1.2, deg2rad(15))

    @info "Smart Cone: Center $(round(rad2deg(center_ang), digits=1))°, Width +/- $(round(rad2deg(half_width), digits=1))°"

    # 4. Generate Rays (Gaussian distribution centered on target)
    return center_ang .+ randn(T, nrays) .* half_width
end


"""
    fit!(model::PlaneWaveCurvModel, train_locs, train_meas;
         init_angles=:auto, source_depth=0.0, reinit=false,
         max_epochs=5000, learning_rate=0.01, verbose=true,
         log_interval=500, min_curvature=50.0, alpha=1e-4)

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
- `min_curvature`: Minimum absolute curvature value for stability
- `alpha`: L1 regularization strength

# Returns
- `loss_history`: Vector of loss values at each logging interval
"""
function fit!(model::PlaneWaveCurvModel, train_locs, train_meas;
              init_angles=:auto,
              source_depth=0.0,
              reinit=false,
              max_epochs=5000,
              learning_rate=0.01,
              verbose=true,
              log_interval=500,
              min_curvature=50.0,
              alpha=1e-4)

    T = eltype(model.theta)
    nrays = model.nrays
    needs_init = all(model.theta .== 0) || reinit

    if needs_init
        verbose && @info "Initializing parameters..."
        model.theta .= initialize_angles(model.env, nrays, init_angles, T; source_depth=source_depth)
        model.A .= rand(T, nrays) .* T(0.1)
        model.phi .= zeros(T, nrays)
    end

    # --- THE PAPER'S LOSS FUNCTION ---
    # 1. MSE on Amplitude (Linear, normalized 0-1)
    # 2. L1 Penalty on Amplitudes (Enforces Sparsity)
    loss_fn(pred, target) = begin
        mse_term = Flux.mse(abs.(pred), abs.(target))
        l1_term = alpha * sum(abs, model.A)
        return mse_term + l1_term
    end

    opt = Flux.Adam(learning_rate)
    ps = Flux.params(model)
    target_amp = abs.(vec(train_meas))
    k = 2π * model.env.frequency / model.env.soundspeed

    # Training loop
    loss_history = Float64[]

    for epoch in 1:max_epochs
        # Optional: Decay LR for fine-tuning
        if epoch == 3000
             opt.eta *= 0.1
             verbose && println("   [Scheduler] Dropping LR to $(opt.eta)")
        end

        grads = Flux.gradient(ps) do
            preds = calculate_field(model, train_locs, k)
            loss_fn(preds, target_amp)
        end
        Flux.update!(opt, ps, grads)

        # Clamp Curvature
        for i in 1:model.nrays
            if abs(model.d[i]) < min_curvature
                model.d[i] = sign(model.d[i]) * min_curvature
            end
        end

        if verbose && (epoch % log_interval == 0)
            preds = calculate_field(model, train_locs, k)
            curr_mse = Flux.mse(abs.(preds), target_amp)
            l1_val = sum(abs, model.A)

            println("   Epoch $epoch: MSE = $(round(curr_mse, digits=6)) | L1 Sum = $(round(l1_val, digits=4))")
            push!(loss_history, curr_mse)
        end
    end

    verbose && println("   Training complete!")
    return loss_history
end


"""
    stack_coordinates(r, z)

Stack range and depth coordinates into a 3×N array for model input.
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
"""
function prepare_measurements(p, scale_factor=1e6)
    return reshape(p .* scale_factor, 1, :)
end
