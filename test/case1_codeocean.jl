using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using UnderwaterAcoustics: SandyClay, SampledField, PressureReleaseBoundary
using AcousticsToolbox
using Flux
using Statistics
using Random
using CSV
using DataFrames
using BSON
using DSP: amp2db


# ============================================================================
# TEST HELPER FUNCTIONS
# ============================================================================

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
    xₒ = [0.0, 0.0] # discrepancy with original code
    x = @view xy[1:1, :]
    y = @view xy[2:2, :]
    xx = x .- (xₒ[1] .- r.d .* cos.(r.θ))
    yy = y .- (xₒ[2] .- r.d .* sin.(r.θ))
    l = sqrt.(xx.^2 + yy.^2)
    kxcys = r.k .* l .+ r.ϕ
    real_im_amp = r.A .* cis.(kxcys)
    amp2db.(abs.(sum(real_im_amp; dims = 1)))
end

function train_model!(model, loss_func, data_loss_func, rx_train, rx_val, TL_train, TL_val; initial_lr = 0.05f0, threshold_count = 5000, threshold_lr = 1e-6, show = false)
    best_model = [deepcopy(p) for p in Flux.params(model)]
    best_loss = data_loss_func(rx_val, TL_val)
    count = 0
    opt = Flux.Adam(initial_lr)

    println("=== CHECKPOINT 2: INITIAL LOSS ===")
    println("Pre-train Train Loss: ", data_loss_func(rx_train, TL_train))
    println("Pre-train Val Loss: ", data_loss_func(rx_val, TL_val))

    # ONLY ONE LOOP!
    for epoch in 1:10_000_000_000
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
            for (p, b) in zip(Flux.params(model), best_model)
                p .= b
            end
            opt.eta /= 10.0f0
            opt.eta < threshold_lr && break
        end
    end
    return model
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

function data_split(rx; ratio = 0.7f0)
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

# ============================================================================
# END TEST HELPER FUNCTIONS
# ============================================================================

@testset "Case 1: Range-Dependent Bathymetry (Bellhop)" begin

    c = 1541.0f0
    f = 10000.0f0
    k = 2.0f0 * Float32(π) * f / c
    L = 30.0f0
    tx = [0.0f0, 5.0f0]
    xmin = 1000.0f0
    xrange = 50.0f0
    zmin = 0.0f0
    zrange = L
    n_rays = 60

    env = UnderwaterEnvironment(
        surface = PressureReleaseBoundary,
        seabed = SandyClay,
        soundspeed = c,
        bathymetry = SampledField([40.0f0, 30.0f0, 33.0f0]; x = [0.0f0, 550.0f0, 1100.0f0])
    )
    pm = AcousticsToolbox.Bellhop(env)

    rx = zig_zag_samples(xmin, xrange, 0.05f0, 1.0f0, 29.0f0, 0.5f0)
    rx_train, rx_val = data_split(rx)

    TL_data = CSV.read("src/data/A_train.csv", DataFrame, header = false, types = Float32) |> Matrix
    TL_train = TL_data[1:size(rx_train, 2)]'
    TL_val = TL_data[1+size(rx_train, 2):end]'



    rx_test, TL_test = generate_test_data(pm, tx, f, xmin, xrange, 0.05f0, zmin, zrange, 0.05f0)

    rbnn = RayBasis(n_rays, k)

    bson_path = "src/bson_logs/ini_RBNN.bson"
    if isfile(bson_path)
        println("\n✅ Loading initial weights from $bson_path...")
        data = BSON.load(bson_path)
        if haskey(data, :rbnn)
            rbnn = data[:rbnn]
            println("=== CHECKPOINT 1: INITIAL WEIGHTS ===")
            println("Sum of θ: ", sum(Flux.params(rbnn)[1]))
            println("Sum of A: ", sum(Flux.params(rbnn)[2]))
            println("Sum of ϕ: ", sum(Flux.params(rbnn)[3]))
            println("Sum of d: ", sum(Flux.params(rbnn)[4]))
        end
    end


    data_loss_rbnn(x, y) = (Flux.Losses.mse(rbnn(x), y))^0.5f0



    rbnn = train_model!(rbnn, data_loss_rbnn, data_loss_rbnn, rx_train, rx_val, TL_train, TL_val; initial_lr = 0.5f0, show = true)

    test_loss = data_loss_rbnn(rx_test, TL_test)
    println("\n=========================================")
    println("FINAL TEST LOSS (RMSE): ", test_loss)
    println("=========================================\n")
    @show test_loss
    # @test isfinite(test_loss)
end
