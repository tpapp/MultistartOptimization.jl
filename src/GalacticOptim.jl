####
#### Local minimization with GalacticOptim. Loaded on demand when `GalacticOptim` is.
####

import GalacticOptim

import MultistartOptimization: local_minimization

export GalacticOptimLocalMethod

Base.@kwdef struct GalacticOptimLocalMethod
	algorithm
	maxiters = nothing
	kwargs = nothing
end

"""
$(SIGNATURES)

A wrapper for algorithms supported by `GalacticOptim`. Used to perform a local
search on the corresponding global optimization problem.

See the GalacticOptim documentation for the options. Defaults aren't changed.
"""
function GalacticOptimLocalMethod(algorithm; maxiters = nothing, kwargs...)
	GalacticOptimLocalMethod(algorithm, maxiters, kwargs)
end

function local_minimization(local_method::GalacticOptimLocalMethod,
                            minimization_problem::MinimizationProblem,
							x)

	@unpack objective, lower_bounds, upper_bounds = minimization_problem
	@unpack algorithm, maxiters, kwargs = local_method

	f(x, p = nothing) = objective(x)
	prob = GalacticOptim.OptimizationProblem(f, x , nothing)

	if typeof(algorithm) == TikTak
		throw(DomainError("Cannot use global optimization algorithm as a local search algorithm"))
	end

	result = GalacticOptim.solve(prob, algorithm(), maxiters = maxiters, kwargs...)

	# check boundary constraints and return `nothing` if any constraint fails
	if any(x -> x == 0, lower_bounds .<= result.minimizer)
		return nothing
	elseif any(x -> x == 0, result.minimizer .<= upper_bounds)
		return nothing
	end

    LocationValue(result.minimizer, result.minimum)
end
