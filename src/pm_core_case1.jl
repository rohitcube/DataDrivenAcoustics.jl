# ============================================================================
# Case 1: PlaneWaveCurvModel — field calculation, training, and utilities
# ============================================================================

"""
    calculate_field(model, coord, k)

RayBasis-style forward model used in the Capsule:
- Input coords are 2×N with x (range) and z (depth, positive).
- Output is transmission loss in dB.
"""
function calculate_field(model::PlaneWaveCurvModel, coord::AbstractArray, k::Number)
    x = @view coord[1:1, :]
    y = @view coord[2:2, :]
    T = eltype(coord)
    xₒ = (T(0), T(0))

    xx = x .- (xₒ[1] .- model.d .* cos.(model.theta))
    yy = y .- (xₒ[2] .- model.d .* sin.(model.theta))
    l = sqrt.(xx.^2 + yy.^2)
    kx = k .* l .+ model.phi
    ray_field = model.A .* cis.(kx)

    return amp2db.(abs.(sum(ray_field; dims = 1)))
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
function fit!(model::PlaneWaveCurvModel, rx_train, rx_val, TL_train, TL_val;
              reinit=false,
              initial_lr=0.5f0,
              threshold_count=5000,
              threshold_lr=1e-6,
              show=false,
              max_epochs=10_000_000_000)

    T = eltype(model.theta)
    nrays = model.nrays

    needs_init = all(model.theta .== 0) || reinit
    if needs_init
        model.theta .= rand(T, nrays) .* T(π)
        model.A .= rand(T, nrays)
        model.phi .= rand(T, nrays) .* T(π)
        model.d .= rand(T, nrays)
    end

    k = T(2) * T(π) * model.env.frequency / model.env.soundspeed

    loss_func(x, y) = (Flux.Losses.mse(calculate_field(model, x, k), y))^0.5f0
    data_loss_func(x, y) = (Flux.Losses.mse(calculate_field(model, x, k), y))^0.5f0

    best_model = [copy(p) for p in Flux.params(model)]
    best_loss = data_loss_func(rx_val, TL_val)
    count = 0
    opt = Flux.Adam(initial_lr)

    for epoch in 1:max_epochs
        Flux.train!(loss_func, Flux.params(model), [(rx_train, TL_train)], opt)
        tmploss = data_loss_func(rx_val, TL_val)

        if best_loss > tmploss
            best_loss = tmploss
            best_model = [deepcopy(p) for p in Flux.params(model)]
            count = 0
            if show
                @show epoch, data_loss_func(rx_train, TL_train), data_loss_func(rx_val, TL_val)
            end
        else
            count += 1
        end

        if count > threshold_count
            count = 0
            # Manually copy the best weights back into the model
            for (p, b) in zip(Flux.params(model), best_model)
                p .= b
            end
            opt.eta /= 10.0f0
            opt.eta < threshold_lr && break
        end
    end

    return model
end


"""
    stack_coordinates(r, z)

Stack range and depth coordinates into a 3×N array for model input.
"""
function stack_coordinates(x, z)
    locs = zeros(eltype(x), 2, length(x))
    locs[1, :] = x
    locs[2, :] = z
    return locs
end


"""
    prepare_measurements(p, scale_factor=1e6)

Prepare complex pressure measurements for model training.
"""
function prepare_measurements(p, scale_factor=1e6)
    return reshape(p .* scale_factor, 1, :)
end
