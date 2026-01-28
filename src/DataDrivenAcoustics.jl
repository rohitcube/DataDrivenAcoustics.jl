module DataDrivenAcoustics

using UnderwaterAcoustics
using DocStringExtensions
using DSP: amp2db, db2amp, pow2db, db2pow
export BasicDataDrivenUnderwaterEnvironment, DataDrivenUnderwaterEnvironment
export RayBasisNN, SphericalWaveModel, plane_wave_propagate, PlaneWaveCurvModel
export fit!, calculate_field

include("pm_core.jl")
include("pm_utility.jl")
include("pm_RBNN.jl")
include("pm_GPR.jl")


#= function __init__()
    UnderwaterAcoustics.addmodel!(RayBasis2D)
    UnderwaterAcoustics.addmodel!(RayBasis2DCurv)
    UnderwaterAcoustics.addmodel!(RayBasis3D)
    UnderwaterAcoustics.addmodel!(RayBasis3DRCNN)
    UnderwaterAcoustics.addmodel!(GPR)
end =#

end