module MultistartOptimization

# NOTE: exports in included files

using ArgCheck: @argcheck
using Base.Threads: @spawn, fetch
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using Parameters: @unpack
using Requires: @require
using Sobol: SobolSeq, Sobol
using GalacticOptim: solve, OptimizationProblem

include("generic_api.jl")
include("tiktak.jl")

function __init__()
    @require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" include("nlopt.jl")
end

####
#### Integrating with GalacticOptim.jl
####

"""
$(TYPEDEF)

Wrapper for a minimization problem when using GalacticOptim.

# Fields

$(FIELDS)
"""
struct GalacticOptimMinimizationProblem{P}
	"The MinimizationProblem to be solved."
	f::MinimizationProblem
	"Parameters for the function (f.objective) to be minimized."
	p::P
end

"""
$(TYPEDEF)

Algorithm and parameters for the `GalacticOptimMinimizationProblem` when using GalacticOptim.
"""
Base.@kwdef struct GalacticOptimLocalMethod
	algorithm

	data = nothing
	maxiters = nothing
	cb = (args...) -> (false)
	progress = false
	kwargs = nothing
end

"""
$(SIGNATURES)

A wrapper for algorithms supported by `GalacticOptim`. Used to construct the corresponding
optimization problem.

See the GalacticOptim documentation for the options. Defaults aren't changed.
"""

function GalacticOptimLocalMethod(algorithm, options)
	GalacticOptimLocalMethod(algorithm = algorthm, options...)
end

"""
$(SIGNATURES)
Solve `minimization_problem` using `local_method` from GalacticOptim, starting from `x`. Return a
`LocationValue`.
"""
function local_minimization(local_method::GalacticOptimLocalMethod,
                            minimization_problem::GalacticOptimMinimizationProblem,
							x)

	@unpack f, p = minimization_problem
	@unpack objective, lower_bounds, upper_bounds = f
	@unpack algorithm, data, maxiters, cb, progress, kwargs = local_method

	prob = OptimizationProblem(objective, x ,p)

	if typeof(algorithm) == NLopt.Opt
		throw(DomainError("Build a problem of type `MinimizationProblem` for using algorithms from NLopt"))
	elseif typeof(algorithm) == TikTak
		throw(DomainError("Cannot use global optimization algorithm as a local search algorithm"))
	end

	result = nothing
	if kwargs == nothing
		if data == nothing
			result = solve(prob, algorithm, maxiters = maxiters, cb = cb, progress = progress)
		else
			result = solve(prob, algorithm, data = data, maxiters = maxiters, cb = cb, progress = progress)
		end
	else
		if data == nothing
			result = solve(prob, algorithm, maxiters = maxiters, cb = cb, progress = progress, kwargs... )
		else
			result = solve(prob, algorithm, data = data, maxiters = maxiters, cb = cb, progress = progress, kwargs... )
		end
	end
    LocationValue(result.minimizer, result.minimum)
end

"""
$(SIGNATURES)

Evaluate and return points of an `N`-element Sobol sequence for a
`GalacticOptimMinimizationProblem` type problem.

When `use_threads`, execution is parallelized using `Threads.@spawn`.
"""
function sobol_starting_points(minimization_problem::GalacticOptimMinimizationProblem, N::Integer,
                               use_threads::Bool)
    @unpack objective, lower_bounds, upper_bounds = minimization_problem.f
    s = SobolSeq(lower_bounds, upper_bounds)
    skip(s, N)                  # better uniformity
    points = Iterators.take(s, N)
    if use_threads
        map(fetch, map(x -> @spawn(LocationValue(x, objective(x,minimization_problem.p))), points))
    else
        map(x -> LocationValue(x, objective(x,minimization_problem.p)), points)
    end
end

end # module
