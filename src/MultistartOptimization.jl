module MultistartOptimization

# NOTE: exports in included files

using ArgCheck: @argcheck
using Base.Threads: @spawn, fetch
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using Sobol: SobolSeq, Sobol

include("generic_api.jl")
include("tiktak.jl")

end # module
