module MultistartOptimization

export MinimizationProblem, NLoptLocalMethod, TikTak, global_minimization, LocationValue

using ArgCheck: @argcheck
using Base.Threads: @spawn, fetch
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using NLopt: NLopt
using Parameters: @unpack
using Sobol: SobolSeq, Sobol

struct LocationValue{T <: AbstractVector{<:Real}, S <: Real}
    location::T
    value::S
end

struct MinimizationProblem{F,T}
    objective::F
    lower_bounds::T
    upper_bounds::T
end

function sobol_starting_points(minimization_problem::MinimizationProblem, N::Integer)
    @unpack objective, lower_bounds, upper_bounds = minimization_problem
    s = SobolSeq(lower_bounds, upper_bounds)
    skip(s, N)                  # better uniformity
    map(fetch, map(x -> @spawn(LocationValue(x, objective(x))), Iterators.take(s, N)))
end

function keep_lowest(xs, N)
    @argcheck 1 ≤ N ≤ length(xs)
    partialsort(xs, 1:N, by = p -> p.value)
end

struct NLoptLocalMethod
    algorithm::NLopt.Algorithm
end

function local_minimization(local_method::NLoptLocalMethod,
                            minimization_problem::MinimizationProblem, x)
    @unpack algorithm = local_method
    @unpack objective, lower_bounds, upper_bounds = minimization_problem
    opt = NLopt.Opt(algorithm, length(x))
    opt.lower_bounds = lower_bounds
    opt.upper_bounds = upper_bounds
    function f̃(x, grad)         # wrapper for NLopt
        @argcheck isempty(grad) # ensure no derivatives are asked for
        objective(x)
    end
    opt.min_objective = f̃
    optf, optx, ret = NLopt.optimize(opt, x)
    # FIXME ret is ignored, save it? is it useful?
    LocationValue(optx, optf)
end

struct TikTak
    quasirandom_N::Int
    initial_N::Int
    θ_min::Float64
    θ_max::Float64
    θ_pow::Float64
end

function TikTak(quasirandom_N; keep_ratio = 0.1, θ_min = 0.1, θ_max = 0.995, θ_pow = 0.5)
    TikTak(quasirandom_N, ceil(keep_ratio * quasirandom_N), θ_min, θ_max, θ_pow)
end

function _weight_parameter(t::TikTak, i)
    @unpack initial_N, θ_min, θ_max, θ_pow = t
    clamp((i / initial_N)^θ_pow, θ_min, θ_max)
end

function tiktak_step(t::TikTak, local_method, minimization_problem,
                     best_point, initial_point, i)
    θ = _weight_parameter(t, i)
    x = @. (1 - θ) * initial_point.location + θ * best_point.location
    local_minimization(local_method, minimization_problem, x)
end

function global_minimization(t::TikTak, local_method, minimization_problem)
    @unpack quasirandom_N, initial_N, θ_min, θ_max, θ_pow = t
    quasirandom_points = sobol_starting_points(minimization_problem, quasirandom_N)
    initial_points = keep_lowest(quasirandom_points, initial_N)
    function _step(best_point, (i, initial_point))
        tiktak_step(t, local_method, minimization_problem, best_point, initial_point, i)
    end
    foldl(_step, enumerate(initial_points); init = first(initial_points))
end

end # module
