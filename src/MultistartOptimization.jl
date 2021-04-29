module MultistartOptimization

# NOTE: exports in included files

using ArgCheck: @argcheck
using Base.Threads: @spawn, fetch
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using Parameters: @unpack
using Requires: @require
using Sobol: SobolSeq, Sobol

include("generic_api.jl")
include("tiktak.jl")

function __init__()
    @require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" include("nlopt.jl")
	@requires GalacticOptim="a75be94c-b780-496d-a8a9-0878b188d577" include("GalacticOptim.jl")

end # module
