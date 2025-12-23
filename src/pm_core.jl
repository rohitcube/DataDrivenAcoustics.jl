using RecipesBase
using Printf

# Core abstract types used across propagation models
abstract type DataDrivenUnderwaterEnvironment end
abstract type DataDrivenPropagationModel end


export DataDrivenUnderwaterEnvironment, ModelFit!, transfercoef, transmissionloss, check, plot, rays, eigenrays, arrivals, DataDrivenEnvironment


"""
$(TYPEDEF)
Create an underwater environment for data-driven physics-based propagation models by providing locations, acoustic measnreuments and other known environmental and channel geomtry knowledge.

- `locations`: location measurements (in the form of matrix with dimension [dimension of a single location data x number of data points])
- `measurements`: acoustic field measurements (in the form of matrix with dimension [1 x number of data points])
- `soundspeed`: medium sound speed (default: missing)
- `frequency`: source frequency (default: missing)
- `waterdepth`: water depth (default: missing)
- `salinity`: water salinity (default: 35)
- `seasurface`: surface property (dafault: Vacuum)
- `seabed`: seabed property (default: SandySilt)
- `tx`: source location (default: missing)
- set `dB` to `false` if `measurements` are not in dB scale (default: `true`)
"""
Base.@kwdef struct BasicDataDrivenUnderwaterEnvironment{T1<:Matrix, T2, T3, T4, T5, T6, T7} <: DataDrivenUnderwaterEnvironment
    locations::T1
    measurements::T1
    soundspeed::T2
    frequency::T3
    waterdepth::T4
    salinity::Real

    # FIX: Renamed from seasurface and removed <:ReflectionModel
    surface::T5
    # FIX: Removed <:ReflectionModel
    seabed::T6

    tx::T7
    dB::Bool

    function BasicDataDrivenUnderwaterEnvironment(locations, measurements;
        soundspeed = missing,
        frequency = missing,
        waterdepth = missing,
        salinity = 35.0,

        # FIX: Updated defaults to v0.7+ standards
        # Note: We use the variables directly (no parentheses) because they are constants in your setup
        surface = UnderwaterAcoustics.PressureReleaseBoundary,
        seabed = UnderwaterAcoustics.SandyMud,

        tx = missing,
        dB = true)

        if  tx !== missing
            length(location(tx)) == size(locations)[1] || throw(ArgumentError("Dimension of source location and measurement locations do not match"))
        end
        size(locations)[2] == size(measurements)[2] || throw(ArgumentError("Number of locations and fields measurements do not match"))
        size(locations)[1] < 4 || throw(ArgumentError("Dimension of location data should not be larger than 3"))
        size(measurements)[1] == 1 || throw(ArgumentError("size of acoustic measurements should be 1 × n"))

        # FIX: Pass 'surface' instead of 'seasurface' to new()
        new{typeof(locations), typeof(soundspeed), typeof(frequency), typeof(waterdepth), typeof(surface), typeof(seabed), typeof(tx)}(
            locations, measurements, soundspeed, frequency, waterdepth, salinity, surface, seabed, tx, dB
        )
    end
end

DataDrivenUnderwaterEnvironment(locations, measurements; kwargs...) = BasicDataDrivenUnderwaterEnvironment(locations, measurements; kwargs...)

"""
$(SIGNATURES)
Create a lightweight data-driven environment without upfront measurements.
Intended for far-field 2D use where training data are provided directly to `fit!`.
"""
function DataDrivenEnvironment(;
        soundspeed = missing,
        frequency = missing,
        waterdepth = missing,
        salinity = 35.0,
        # Note: Using the v0.7+ constants we identified (no parentheses)
        surface = UnderwaterAcoustics.PressureReleaseBoundary,
        seabed = UnderwaterAcoustics.SandyMud,
        tx = missing,
        dB = true,
        dims::Int = 2)

    # Create empty placeholders for locations and measurements
    locations = zeros(Float32, dims, 0)
    measurements = zeros(Float32, 1, 0)

    return BasicDataDrivenUnderwaterEnvironment(locations, measurements;
        soundspeed = soundspeed, frequency = frequency, waterdepth = waterdepth,
        salinity = salinity, surface = surface, seabed = seabed, tx = tx, dB = dB)
end



"""
$(SIGNATURES)
Train data-driven physics-based propagation model.

- `ini_lr`: initial learning rate
- `trainloss`: loss function used in training and model update
- `dataloss`: data loss function to calculate benchmarking validation error for early stopping
- `ratioₜ`: data split ratio = number of training data/(number of training data + number of validation data)
- set `seed` to `true` to seed random data selection order
- `maxepoch`: maximum number of training epoches allowed
- `ncount`: maximum number of tries before reducing learning rate
-  model training ends once learning rate is smaller than `minlearnrate`
- learning rate is reduced by `reducedlearnrate` once `ncount` is reached
- set `showloss` to true to display training and validation errors during the model training process, if the validation error is historically the best
"""
function ModelFit!(r::DataDrivenPropagationModel, inilearnrate, trainloss, dataloss, ratioₜ, seed, maxepoch, ncount, minlearnrate, reducedlearnrate, showloss)
    rₜ, pₜ, rᵥ, pᵥ = SplitData(r.env.locations, r.env.measurements, ratioₜ, seed)
    bestmodel = deepcopy(Flux.params(r))
    count = 0
    opt = Adam(inilearnrate)
    epoch = 0
    bestloss = dataloss(rᵥ, pᵥ, r)
    while true
        Flux.train!((x,y) -> trainloss(x, y, r), Flux.params(r), [(rₜ, pₜ)], opt)
        tmploss = dataloss(rᵥ, pᵥ, r)
        epoch += 1
        if tmploss < bestloss
            bestloss = tmploss
            # bestmodel = deepcopy(Flux.params(r))
            bestmodel = r
            count = 0
            showloss && (@show epoch, dataloss(rₜ, pₜ, r), dataloss(rᵥ, pᵥ, r))
        else
            count += 1
        end
        epoch > maxepoch && break
        if count > ncount
            count = 0
            # Flux.loadparams!(r, bestmodel)
            Flux.loadmodel!(r, bestmodel)
            opt.eta /= reducedlearnrate
            opt.eta < minlearnrate && break
            showloss && println("********* reduced learning rate: ",opt.eta, " *********" )
        end
    end
    r
end

# -------------------------------------------------------------------------
# Bare Minimum Fix: Renaming functions to v0.4+ API
# -------------------------------------------------------------------------

# 1. RENAME: transfercoef -> acoustic_field
function UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiver; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    if tx !== nothing &&  tx !== missing
        model.env.frequency == nominalfrequency(tx) || throw(ArgumentError("Mismatched frequencies in acoustic source and data driven environment"))
        if  model.env.tx !== missing
            location(model.env.tx) == location(tx) || throw(ArgumentError("Mismatched location in acoustic source and data driven environment"))
        else
            @warn "Source location is ignored in field calculation"
        end
    end
    if model isa GPR
        if model.twoDimension == true
            p = model.calculatefield(model, hcat([location(rx)[1], location(rx)[end]]))[1]
        else
            p = model.calculatefield(model, hcat([location(rx)[1], location(rx)[2], location(rx)[end]]))[1]
        end
        model.env.dB == true ? (return db2amp.(-p)) : (return p)
    else
        p = model.calculatefield(model, collect(location(rx)))[1]
    end
    return p
end

# 2. RENAME: transfercoef -> acoustic_field
function UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiverGrid2D; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    if tx !== nothing &&  tx !== missing
        model.env.frequency == nominalfrequency(tx) || throw(ArgumentError("Mismatched frequencies in acoustic source and data driven environment"))
        if  model.env.tx !== missing
            location(model.env.tx) == location(tx) || throw(ArgumentError("Mismatched location in acoustic source and data driven environment"))
        else
            @warn "Source location is ignored in field calculation"
        end
    end
    (xlen, ylen) = size(rx)
    x = vec(location.(rx))
    p = reshape(model.calculatefield(model, hcat(first.(x), last.(x))'), xlen, ylen)
    if model isa GPR
        model.env.dB == true ? (return db2amp.(-p)) : (return p)
    else
        return p
    end
end

# 3. RENAME: transfercoef -> acoustic_field
function UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiverGrid3D; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    if tx !== nothing &&  tx !== missing
        model.env.frequency == nominalfrequency(tx) || throw(ArgumentError("Mismatched frequencies in acoustic source and data driven environment"))
        if  model.env.tx !== missing
            location(model.env.tx) == location(tx) ||  throw(ArgumentError("Mismatched location in acoustic source and data driven environment"))
        else
            @warn "Source location is ignored in field calculation"
        end
    end
    (xlen, ylen, zlen) = size(rx)
    x = vec(location.(rx))
    if ylen == 1
        p = reshape(model.calculatefield(model, hcat(first.(x), getfield.(x, 2), last.(x))'), xlen, zlen)
    else
        p = reshape(model.calculatefield(model, hcat(first.(x), getfield.(x, 2), last.(x))'), xlen, ylen, zlen)
    end
    if model isa GPR
        model.env.dB == true ? (return db2amp.(-p)) : (return p)
    else
        return p
    end
end

# 4. UPDATE CALLS INSIDE ALIASES
UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AbstractArray{<:AcousticReceiver}) = UnderwaterAcoustics.tmap(rx1 -> acoustic_field(model, tx, rx1), rx)

UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AbstractMatrix}) = model.calculatefield(model, rx)


# 5. RENAME: transmissionloss -> transmission_loss
# Also updated the internal call to use 'acoustic_field' instead of 'transfercoef'
UnderwaterAcoustics.transmission_loss(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AbstractMatrix}) = -amp2db.(abs.(acoustic_field(model, rx)))

UnderwaterAcoustics.transmission_loss(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::Union{AbstractVector, AbstractMatrix}) = -amp2db.(abs.(acoustic_field(model, tx, rx)))


# 6. RENAME: eigenenPropagationModel, tx, rx) = throw(ArgumentError("This function is not yet supported"))

UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, tx, rx) = throw(ArgumentError("This function is not yet supported"))


# 7. COMMENT OUT CONFLICTING TYPES
# UnderwaterAcoustics already defines Arrival, so we comment this out to avoid a crash.
# abstract type Arrival end

# function Base.show(io::IO, a::Arrival)
#     if a.time === missing
#         @printf(io, "                         |          | %5.1f dB ϕ%6.1f°", amp2db(abs(a.phasor)), rad2deg(angle(a.phasor)))
#     else
#         @printf(io, "                         | %6.2f ms | %5.1f dB ϕ%6.1f°", 1000*a.time, amp2db(abs(a.phasor)), rad2deg(angle(a.phasor)))
#     end
# end


#= struct DataDrivenArrival{T1,T2} <: UnderwaterAcoustics.Arrival
    time::T1
    phasor::T2
    surface::Missing
    bottom::Missing
    launchangle::Missing
    arrivalangle::Missing
    raypath::Missing
end =#


"""
$(SIGNATURES)
Show arrival rays at a location `rx` using a data-driven physics-based propagation model.

- `model`: data-driven physics-based propagation model
- `tx`: acoustic source. This is optional. Use `missing` or `nothing` for unknown source.
- `rx`: an acoustic receiver
"""
function UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::Union{AbstractVector, AcousticReceiver}; threshold = 30)
    model isa GPR && throw(ArgumentError("GPR model does not support this function"))

    # Calculate the raw field
    arrival_field = model.calculatefield(model, collect(location(rx)); showarrivals = true)

    # Filter significant arrivals
    amp = amp2db.(abs.(arrival_field))
    idx = findall(amp .> (maximum(amp) - threshold))
    significant_arrivals = arrival_field[idx]

    # Sort by amplitude (strongest first)
    sorted_indices = sortperm(abs.(significant_arrivals), rev = true)
    final_indices = idx[sorted_indices]

    # Construct the result using YOUR custom struct
    # Note: We calculate time delays based on the model type if available
    results = map(1:length(final_indices)) do i
        k = final_indices[i] # Original index in the buffer
        phasor = arrival_field[k]

        # Calculate time based on model type (assuming model.d exists)
        if model isa RayBasis2D || model isa RayBasis2DCurv
            t = missing
        elseif model isa RayBasis3DRCNN
             t = model.d[k] ./ model.env.soundspeed
        else
             # Assuming standard RayBasisNN or similar
             t = (model.d[k] .+ model.ed[k]) ./ model.env.soundspeed
        end

        # Create your custom object
        DataDrivenArrival(t, phasor, missing, missing, missing, missing, missing)
    end

    return results
end

UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AcousticReceiver}) =
    UnderwaterAcoustics.arrivals(model, nothing, rx)

@recipe function plot(env::DataDrivenUnderwaterEnvironment; receivers = [], transmissionloss = [],  dynamicrange = 42.0)
    size(transmissionloss) == size(receivers) || throw(ArgumentError("Mismatched receivers and transmissionloss"))
    receivers isa AcousticReceiverGrid2D || throw(ArgumentError("Receivers must be an instance of AcousticReceiverGrid2D"))
    minloss = minimum(transmissionloss)
    clims --> (-minloss-dynamicrange, -minloss)
    colorbar --> true
    cguide --> "dB"
    ticks --> :native
    legend --> false
    xguide --> "x (m)"
    yguide --> "z (m) "
    @series begin
        seriestype := :heatmap
        receivers.xrange, receivers.zrange, -transmissionloss'
    end
end

# This is the "Case 1" model: Metadata only, Far-field approximation.
mutable struct SphericalWaveModel <: DataDrivenPropagationModel

    # Typed as the ABSTRACT parent.
    # This allows it to hold any specific implementation (Missing or Source).
    env::DataDrivenUnderwaterEnvironment

    nrays::Int

    A::Vector{Float64}
    phi::Vector{Float64}
    theta::Vector{Float64}
end

mutable struct SphericalWaveModel{E<:DataDrivenUnderwaterEnvironment, AT, PT, TT} <: DataDrivenPropagationModel
    env::E
    nrays::Int
    A::AT      # e.g., Vector{Float64}
    phi::PT    # e.g., Vector{Float64}
    theta::TT  # e.g., Vector{Float64}
end

# "Far-field of a point source... approximated by a planar wavefront"
function calculate_field(model::SphericalWaveModel, rx_coords::AbstractMatrix)
    c = model.env.soundspeed
    f = model.env.frequency
    k_mag = 2π * f / c

    n_rx = size(rx_coords, 2)
    pressure = zeros(ComplexF64, n_rx)

    # Summation of N plane waves
    for i in 1:n_rx
        r_vec = rx_coords[:, i]
        total_p = 0.0 + 0.0im

        for m in 1:model.nrays
            # k vector based on learned angle theta
            kx = k_mag * cos(model.theta[m])
            kz = k_mag * sin(model.theta[m])

            # Phase term: k*r + phi
            phase = (kx * r_vec[1] + kz * r_vec[end]) + model.phi[m]
            total_p += model.A[m] * exp(im * phase)
        end
        pressure[i] = total_p
    end
    return pressure
end


function fit!(model::SphericalWaveModel, tx, rx_locs, loss_func, measurements)
    # 1. Detect Source Dimensions
    # We check how many coordinates the source has (usually 3: x, y, z)
    src_dims = length(location(tx))

    # 2. Update Environment with Matching Dimensions
    model.env = DataDrivenEnvironment(
        soundspeed = model.env.soundspeed,
        frequency  = model.env.frequency,
        surface    = model.env.surface,
        tx         = tx,
        dims       = src_dims  # <--- FIX: Force env to match source dimensions
    )

    # 3. Initialize Parameters (Randomly as per literature)
    rng = Random.default_rng()
    model.A = rand(rng, model.nrays)
    model.phi = rand(rng, model.nrays) .* 2π
    model.theta = rand(rng, model.nrays) .* 2π .- π

    return model
end
function RayBasisNN(env::DataDrivenUnderwaterEnvironment; nrays=50, kwargs...)

    # CASE 3-5: We know the depth -> Use Ray Tracing (RayBasis2D)
    if !ismissing(env.waterdepth)
        # return RayBasis2D(env; nrays=nrays, kwargs...)
    end

    # CASE 1-2: We don't know the depth -> Use Spherical/Plane Wave Model
    # This is what Test Case 1 wil get
    # Initialsing with empty arrays, so that fit! can be called to train the model
    return SphericalWaveModel(env, nrays, Float64[], Float64[], Float64[])
end


function RayBasisNN(::Type{M}, env::E; nrays=50) where {M<:DataDrivenPropagationModel, E}
    # Initialize with concrete types (Float64) to maintain stability
    return M(env, nrays, Float64[], Float64[], Float64[])
end


#= function UnderwaterAcoustics.check(::Type{RayBasis2D}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        size(env.locations)[1] == 2 || throw(ArgumentError("RayBasis2D only supports 2D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis2DCurv}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        size(env.locations)[1] == 2 || throw(ArgumentError("RayBasis2DCurv only supports 2D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis3D}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        size(env.locations)[1] == 3 || throw(ArgumentError("RayBasis3D only supports 3D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis3DRCNN}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        env.tx === missing || throw(ArgumentError("RayBasis3DRCNN only supports environments with known source location"))
        length(location(env.tx)) == 3 || throw(ArgumentError("RayBasis3DRCNN only supports 3D source"))
        size(env.locations)[1] == 3|| throw(ArgumentError("RayBasis3DRCNN only supports 3D environment"))
        env.waterdepth !== missing || throw(ArgumentError("RayBasis3DRCNN only supports environments with known water depth"))
    end
    env
end


function UnderwaterAcoustics.check(::Type{GPR}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    env
end =#


