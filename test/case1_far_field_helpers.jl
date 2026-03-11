using UnderwaterAcoustics
using AcousticsToolbox
using Flux
using Random
using DSP: amp2db

import Flux.Optimise: apply!, AbstractOptimiser

mutable struct LegacyADAM <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64, Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end

LegacyADAM(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8) = LegacyADAM(η, β, ϵ, IdDict())

function apply!(o::LegacyADAM, x, Δ)
  η, β, ϵ = o.eta, o.beta, o.epsilon
  mt, vt, βp = get!(o.state, x) do
    (zero(x), zero(x), Float64[β[1], β[2]])
  end
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
  @. Δ =  mt / (1 - βp[1]) / (sqrt(vt / (1 - βp[2])) + ϵ) * η
  βp .= βp .* β
  return Δ
end

function fit_case1_train!(model, rx_train, rx_val, TL_train, TL_val;
              initial_lr = Float32(0.5),
              threshold_count = 5000,
              threshold_lr = Float32(1e-6),
              show = false)
    data_loss(x, y) = (Flux.Losses.mse(model(x), y))^0.5f0
    loss_func = data_loss

    best_model = [deepcopy(p) for p in Flux.params(model)]
    best_loss = data_loss(rx_val, TL_val)
    count = 0
    opt = LegacyADAM(initial_lr)

    println("=== CHECKPOINT 2: INITIAL LOSS ===")
    println("Pre-train Train Loss: ", data_loss(rx_train, TL_train))
    println("Pre-train Val Loss: ", data_loss(rx_val, TL_val))

    println("=== CHECKPOINT 2.5: GRADIENT CHECK ===")
    ps = Flux.params(model)

    grads = Flux.gradient(ps) do
        loss_func(rx_train, TL_train)
    end

    for (name, param) in [("A", model.A), ("θ", model.θ), ("d", model.d), ("ϕ", model.ϕ)]
        grad_p = grads[param]
        println("--- Parameter $name ---")

        if grad_p === nothing
            println("Type: Nothing (Gradient not tracked or disconnected!)")
        else
            println("Type: ", typeof(grad_p))
            n_items = min(length(grad_p), 3)
            println("Gradient (first $n_items): ", grad_p[1:n_items])
        end
    end
    println("============================================")

    A_prev = copy(model.A)

    for epoch in 1:10_000_000_000
        Flux.train!(loss_func, Flux.params(model), [(rx_train, TL_train)], opt)
        if epoch == 1
            println("=== CHECKPOINT 3: AFTER EPOCH 1 ===")
            println("New Sum of A: ", sum(Flux.params(model)[1]))
            println("New Train Loss: ", data_loss(rx_train, TL_train))
            mt, vt, βp = opt.state[model.A]

            println("--- OPTIMIZER STATE LOG (Epoch 1) ---")
            println("Sum of Momentum (mt) for A: ", sum(mt))
            println("Sum of Velocity (vt) for A: ", sum(vt))
            println("Current Beta Power (βp): ", βp)
            println("-------------------------------------")
        end

        if epoch % 1000 == 0
            standard_sum_A = sum(model.A)
            current_movement = sum(abs.(model.A .- A_prev))

            println("--- Epoch $epoch Summary ---")
            println("Current Sum of A: $standard_sum_A")
            println("Movement since last check: $current_movement")

            A_prev = copy(model.A)
        end

        tmploss = data_loss(rx_val, TL_val)
        if best_loss > tmploss
            best_loss = tmploss
            best_model = [deepcopy(p) for p in Flux.params(model)]
            count = 0
            if show
                @show epoch, data_loss(rx_train, TL_train), data_loss(rx_val, TL_val)
            end
        else
            count += 1
        end

        if count > threshold_count
            println(">>> EVENT: LR DROP at Epoch $epoch")
            println(">>> Sum of A at failure: ", sum(model.A))
            println(">>> Loss was stuck at: $best_loss")

            count = 0
            for (p, b) in zip(Flux.params(model), best_model)
                p .= b
            end
            opt.eta /= 10.0
            A_prev = copy(model.A)
            opt.eta < threshold_lr && break
        end
    end

    return model
end

function fit!(env, tx, rx, tl_fn, data;
              model = nothing,
              nrays = 60,
              initial_lr = Float32(0.5),
              threshold_count = 5000,
              threshold_lr = Float32(1e-6),
              show = false)
    rx_train, rx_val = data_split(rx)

    TL_train = data[1:size(rx_train, 2)]'
    TL_val = data[1+size(rx_train, 2):end]'

    if model === nothing
        k = Float32(2.0) * π * env.frequency / env.soundspeed
        model = RayBasis2DCurv(nrays, k)
    end

    return fit_case1_train!(model, rx_train, rx_val, TL_train, TL_val;
        initial_lr = initial_lr,
        threshold_count = threshold_count,
        threshold_lr = threshold_lr,
        show = show
    )
end

struct RayBasis{T1<:AbstractVector,T2<:Real}
    θ::T1
    A::T1
    ϕ::T1
    d::T1
    k::T2
end

RayBasis(rays::Integer, k::Real) = RayBasis(rand(Float32, rays) * π, rand(Float32, rays), rand(Float32, rays) * π, rand(Float32, rays), k)

Flux.@functor RayBasis
Flux.trainable(r::RayBasis) = (r.θ, r.A, r.ϕ, r.d)

function (r::RayBasis)(xy::AbstractArray)
    xₒ = [0.0, 0.0]
    x = @view xy[1:1, :]
    y = @view xy[2:2, :]
    xx = x .- (xₒ[1] .- r.d .* cos.(r.θ))
    yy = y .- (xₒ[2] .- r.d .* sin.(r.θ))
    l = sqrt.(xx.^2 + yy.^2)
    kxcys = r.k .* l .+ r.ϕ
    real_im_amp = r.A .* cis.(kxcys)
    amp2db.(abs.(sum(real_im_amp; dims = 1)))
end

struct RayBasis2DCurv{T1<:AbstractVector,T2<:Real}
    θ::T1
    A::T1
    ϕ::T1
    d::T1
    k::T2
end

RayBasis2DCurv(rays::Integer, k::Real) = RayBasis2DCurv(rand(Float32, rays) * π, rand(Float32, rays), rand(Float32, rays) * π, rand(Float32, rays), k)

Flux.@functor RayBasis2DCurv
Flux.trainable(r::RayBasis2DCurv) = (r.θ, r.A, r.ϕ, r.d)

function (r::RayBasis2DCurv)(xy::AbstractArray)
    xₒ = [0.0, 0.0]
    x = @view xy[1:1, :]
    y = @view xy[2:2, :]
    xx = x .- (xₒ[1] .- r.d .* cos.(r.θ))
    yy = y .- (xₒ[2] .- r.d .* sin.(r.θ))
    l = sqrt.(xx.^2 + yy.^2)
    kxcys = r.k .* l .+ r.ϕ
    real_im_amp = r.A .* cis.(kxcys)
    amp2db.(abs.(sum(real_im_amp; dims = 1)))
end

function remove_consecutive_duplicates(v)
    out = Vector{eltype(v)}()
    for item in v
        if isempty(out) || out[end] != item
            push!(out, item)
        end
    end
    return out
end

function zig_zag_samples(xmin, xrange, xscale, zmin, zrange, zscale; IsTwoD = true)
    vt = zrange / zscale
    ht = xrange / xscale
    z_in = collect(zmin:zscale:zmin + zrange)
    z_de = collect(zmin + zrange:-zscale:zmin)
    z = Array{Float32}(undef, 0)
    for i in 1:ceil(Int, ht / vt)
        iseven(i) ? (z = vcat(z, z_de)) : (z = vcat(z, z_in))
    end
    z = remove_consecutive_duplicates(z)
    x = collect(xmin:xscale:xmin + xscale * (length(z) - 1))
    idx = findall(x .< (xmin + xrange))
    return IsTwoD == true ? vcat(x', z')[:, idx] : vcat(x', zeros(Float32, 1, length(x)), z')[:, idx]
end

function data_split(rx; ratio = Float32(0.7))
    data_len = size(rx)[2]
    Random.seed!(data_len)
    data_idx = randperm(data_len)
    idx_train = data_idx[1:Int(floor(data_len * ratio))]
    idx_val = data_idx[Int(floor(data_len * ratio)) + 1:end]

    rx_train = rx[:, idx_train]
    rx_val = rx[:, idx_val]

    return rx_train, rx_val
end

function generate_test_data(pm, tx, f, xmin, xrange, xs, zmin, zrange, zs; IsTwoD = true)
    x_range = xmin:xs:xmin + xrange
    z_range = -(zmin + zrange):zs:-zmin
    rx = AcousticReceiverGrid2D(x_range, z_range)
    TL = -transmission_loss(pm, AcousticSource(tx[1], -tx[end], f), rx)

    x = collect(Float32, x_range)'
    z = collect(Float32, zmin + zrange:-zs:zmin)'
    IsTwoD == true ?
        (rx_test = vcat(repeat(x, 1, length(z)), repeat(z, inner = (1, length(x))))) :
        (rx_test = vcat(repeat(x, 1, length(z)), zeros(Float32, 1, length(x) * length(z)), repeat(z, inner = (1, length(x)))))
    return rx_test, reshape(TL, 1, length(TL))
end
