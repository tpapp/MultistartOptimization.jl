####
#### generic API
####

export MinimizationProblem, local_minimization, multistart_minimization, NLopt_local_method

struct MinimizationProblem{F,T<:AbstractVector}
    "The function to be minimized."
    objective::F
    "Lower bounds (a vector of real numbers)."
    lower_bounds::T
    "Upper bounds (a vector of real numbers)."
    upper_bounds::T
    @doc """
    $(SIGNATURES)

    Define a minimization problem.

    - `objective` is a function that maps `ℝᴺ` vectors to real numbers. It should accept all
       `AbstractVector`s of length `N`.
    - `lower_bounds` and `upper_bounds` define the bounds, and their lengths implicitly
      define the dimension `N`.

    The fields of the result should be accessible with the names above.
    """
    function MinimizationProblem(objective::F, lower_bounds::T, upper_bounds::T) where {F,T}
        @argcheck all(lower_bounds .< upper_bounds)
        new{F,T}(objective, lower_bounds, upper_bounds)
    end
end

"""
`$(FUNCTIONNAME)(minimization_problem, local_method, x)`

Perform a local minimization using `local_method`, starting from `x`.

The objective and the bounds are provided in `minimization_problem`, see
[`MinimizationProblem`](@ref). `x` should be an `AbstractVector` of conforming dimension.

`local_method` can be a type for which a method is defined (recommended). However, it can
also be a closure, in which case it will be called as `local_method(minimization_problem, x)`.

In both cases, it should return `nothing`, or a value which has the following properties:

- `location`, an `AbstractVector` for the minimizer.

- `value`, for the value of the objective at `location`. `Inf` (or equivalent) should be
  used for infeasible regions.

The returned value may have other properties too, these are useful for convergence
diagnostics, debugging information, etc, these depend on the `local_method`.

Returning `nothing` is equivalent to `value == Inf`, but in some cases can work better for
type inference as the method won't have to construct a counterfactual type.
"""
function local_minimization(local_method, minimization_problem::MinimizationProblem, x)
    local_method(minimization_problem, x)
end

"""
`$(FUNCTIONNAME)(multistart_method, local_method, minimization_problem)`

Multistart minimization using the given multistart and local methods.
"""
function multistart_minimization end

"""
A placeholder for integrating into the `NLopt` interface.

Methods are added when `NLopt` is loaded (via weak dependencies).
"""
function NLopt_local_method end
