module MultistartOptimization

export MinimizationProblem, LocationValue, NLoptLocalMethod, local_minimization, TikTak,
    multistart_minimization

using ArgCheck: @argcheck
using Base.Threads: @spawn, fetch
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using NLopt: NLopt
using Parameters: @unpack
using Sobol: SobolSeq, Sobol
using ProgressMeter

####
#### structures for the problem and results
####

"""
$(TYPEDEF)

Wrapper for a minimization problem.

# Fields

$(FIELDS)
"""
struct MinimizationProblem{F,T}
    "The function to be minimized."
    objective::F
    "Lower bounds (a vector of real numbers)."
    lower_bounds::T
    "Upper bounds (a vector of real numbers)."
    upper_bounds::T
    # FIXME constructor checks
end

"""
$(TYPEDEF)

A location-value pair.

# Fields

$(FIELDS)
"""
struct LocationValue{T <: AbstractVector{<:Real}, S <: Real}
    "Location (a vector of real numbers)."
    location::T
    "The value of the objective at `location`."
    value::S
end

####
#### internal utilities
####

"""
$(SIGNATURES)

Evaluate and return points of an `N`-element Sobol sequence.

An effort is made to parallelize the code using `Threads` when available.
"""
function sobol_starting_points(minimization_problem::MinimizationProblem, N::Integer; parallel_map=true)
    @unpack objective, lower_bounds, upper_bounds = minimization_problem
    s = SobolSeq(lower_bounds, upper_bounds)
    skip(s, N)                  # better uniformity
    if parallel_map
        map(fetch, map(x -> @spawn(LocationValue(x, objective(x))), Iterators.take(s, N)))
    else
        map(x -> LocationValue(x, objective(x)), Iterators.take(s, N))
    end
end

"""
$(SIGNATURES)

Helper function to keep the `N` points with the lowest `value`.
"""
function _keep_lowest(xs, N)
    @argcheck 1 ≤ N ≤ length(xs)
    partialsort(xs, 1:N, by = p -> p.value)
end

####
#### local minimization
####

Base.@kwdef struct NLoptLocalMethod
    algorithm::NLopt.Algorithm
    xtol_abs::Float64 = 1e-8
    xtol_rel::Float64 = 1e-8
    maxeval::Int = 0
    maxtime::Float64 = 0.0
end

"""
$(SIGNATURES)

A wrapper for algorithms supported by `NLopt`. Used to construct the corresponding
optimization problem.

See the NLopt documentation for the options. Defaults are changed slightly.
"""
function NLoptLocalMethod(algorithm::NLopt.Algorithm; options...)
    NLoptLocalMethod(; algorithm = algorithm, options...)
end

"""
$(SIGNATURES)

Solve `minimization_problem` using `local_method`, starting from `x`. Return a
`LocationValue`.
"""
function local_minimization(local_method::NLoptLocalMethod,
                            minimization_problem::MinimizationProblem, x)
    @unpack algorithm, xtol_abs, xtol_rel, maxeval, maxtime = local_method
    @unpack objective, lower_bounds, upper_bounds = minimization_problem
    opt = NLopt.Opt(algorithm, length(x))
    opt.lower_bounds = lower_bounds
    opt.upper_bounds = upper_bounds
    function f̃(x, grad)         # wrapper for NLopt
        @argcheck isempty(grad) # ensure no derivatives are asked for
        objective(x)
    end
    opt.min_objective = f̃
    opt.xtol_abs = xtol_abs
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval
    opt.maxtime = maxtime
    optf, optx, ret = NLopt.optimize(opt, x)
    # FIXME ret is ignored, save it? is it useful?
    LocationValue(optx, optf)
end

####
#### multistart minimization
####

struct TikTak
    quasirandom_N::Int
    initial_N::Int
    θ_min::Float64
    θ_max::Float64
    θ_pow::Float64
end

"""
$(SIGNATURES)

The “TikTak” multistart method, as described in *Arnoud, Guvenen, and Kleineberg (2019)*.

This implements the *multistart* part, can be called with arbitrary local methods, see
[`multistart_minimization`](@ref).

# Arguments

- `quasirandom_N`: the number of quasirandom points for the first pass (using a Sobol
  sequence).

# Keyword arguments

- `keep_ratio`: the fraction of best quasirandom points which are kept

- `θ_min` and `θ_max` clamp the weight parameter, `θ_pow` determines the power it is raised
  to.

The defaults are from the paper cited above.
"""
function TikTak(quasirandom_N; keep_ratio = 0.1, θ_min = 0.1, θ_max = 0.995, θ_pow = 0.5)
    @argcheck 0 < keep_ratio ≤ 1
    TikTak(quasirandom_N, ceil(keep_ratio * quasirandom_N), θ_min, θ_max, θ_pow)
end

function _weight_parameter(t::TikTak, i)
    @unpack initial_N, θ_min, θ_max, θ_pow = t
    clamp((i / initial_N)^θ_pow, θ_min, θ_max)
end

"""
$(SIGNATURES)

Solve `minimization_problem` by using `local_method` within `multistart_method`.
"""
function multistart_minimization(multistart_method::TikTak, local_method,
                                 minimization_problem; progress=false, parallel_sobol_points=true)
    @unpack quasirandom_N, initial_N, θ_min, θ_max, θ_pow = multistart_method
    quasirandom_points = sobol_starting_points(minimization_problem, quasirandom_N; parallel_map=parallel_sobol_points)
    initial_points = _keep_lowest(quasirandom_points, initial_N)
    if progress
        prog = Progress(length(initial_points))
    end

    function _step(visited_minimum, (i, initial_point))
        θ = _weight_parameter(multistart_method, i)
        x = @. (1 - θ) * initial_point.location + θ * visited_minimum.location
        local_minimum = local_minimization(local_method, minimization_problem, x)
        if @isdefined(prog)
            ProgressMeter.next!(prog)
        end
        local_minimum.value < visited_minimum.value ? local_minimum : visited_minimum
    end
    foldl(_step, enumerate(initial_points); init = first(initial_points))
end


end # module
