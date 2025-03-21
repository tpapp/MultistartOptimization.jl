####
#### Implementation of the TikTak method.
####

export TikTak

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
    (; initial_N, θ_min, θ_max, θ_pow) = t
    clamp((i / initial_N)^θ_pow, θ_min, θ_max)
end

"""
$(SIGNATURES)

Determine if the value of the objective is a finite or `-Inf` real number.

Internal, a basic sanity check to catch errors early.
"""
_acceptable_value(value) = value isa Real && (isfinite(value) || isinf(value)) # we don't want NaNs

"""
$(SIGNATURES)

Evaluate `objective` at `location`, returning `location` and `value` in a `NamedTuple`.
Perform basic sanity checks.

Internal.
"""
function _objective_at_location(objective, location)
        value = objective(location)
        @argcheck _acceptable_value(value)
        (; location, value)
end

"""
$(SIGNATURES)

Evaluate and return points of an `N`-element Sobol sequence.

When `use_threads`, execution is parallelized using `Threads.@spawn`.
"""
function sobol_starting_points(minimization_problem::MinimizationProblem, N::Integer,
                               use_threads::Bool)
    (; objective, lower_bounds, upper_bounds) = minimization_problem
    s = SobolSeq(lower_bounds, upper_bounds)
    skip(s, N)                  # better uniformity
    points = Iterators.take(s, N)
    _initial(x) = _objective_at_location(objective, x)
    if use_threads
        map(fetch, map(x -> @spawn(_initial(x)), points))
    else
        map(_initial, points)
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

"""
$(SIGNATURES)

Solve `minimization_problem` by using `local_method` within `multistart_method`.

When `use_threads`, initial point search is parallelized using `Threads.@spawn`.

`prepend_points` should contain a vector of initial starting points that are prepended to
the Sobol sequence. These are useful if a guess is available for the vicinity of the
optimum.
"""
function multistart_minimization(multistart_method::TikTak, local_method,
                                 minimization_problem;
                                 use_threads = true,
                                 prepend_points = Vector{Vector{Float64}}())
    (; quasirandom_N, initial_N, θ_min, θ_max, θ_pow) = multistart_method
    (; objective, lower_bounds, upper_bounds) = minimization_problem
    @argcheck(all(x -> all(lower_bounds .≤ x .≤ upper_bounds), prepend_points),
              "prepend_points outside problem bounds")
    quasirandom_points = sobol_starting_points(minimization_problem, quasirandom_N,
                                               use_threads)
    initial_points = _keep_lowest(quasirandom_points, initial_N)
    all_points = vcat(map(x -> _objective_at_location(objective, x), prepend_points),
                      initial_points)
    function _step(visited_minimum, (i, initial_point))
        θ = _weight_parameter(multistart_method, i)
        x = @. (1 - θ) * initial_point.location + θ * visited_minimum.location
        local_minimum = local_minimization(local_method, minimization_problem, x)
        local_minimum ≡ nothing && return visited_minimum
        local_minimum.value < visited_minimum.value ? local_minimum : visited_minimum
    end
    foldl(_step, enumerate(Iterators.drop(all_points, 1)); init = first(all_points))
end
