####
#### Local minimization with NLopt. Loaded on demand when `NLopt` is.
####

module MultistartOptimizationNLoptExt

import NLopt

import MultistartOptimization: local_minimization, MinimizationProblem,
    MultistartOptimization, NLopt_local_method

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES

const NLopt_ret_success = Set([:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED,
                               :MAXEVAL_REACHED, :MAXTIME_REACHED])

Base.@kwdef struct NLoptLocalMethod{S}
    algorithm::NLopt.Algorithm
    xtol_abs::Float64
    xtol_rel::Float64
    maxeval::Int
    maxtime::Float64
    "Return values which are considered as “success”."
    ret_success::S
end

"""
$(SIGNATURES)

A wrapper for algorithms supported by `NLopt`. Used to construct the corresponding
optimization problem. All return values in `ret_success` are considered valid (`ret` is also
kept in the result), all negative return values are considered invalid.

See the NLopt documentation for the options. Defaults are changed slightly.
"""
function NLopt_local_method(algorithm::NLopt.Algorithm; xtol_abs = 1e-8, xtol_rel = 1e-8,
                            maxeval = 0, maxtime = 0.0, ret_success = NLopt_ret_success)
    NLoptLocalMethod(; algorithm, xtol_abs, xtol_rel, maxeval, maxtime, ret_success)
end

"""
$(SIGNATURES)

Solve `minimization_problem` using `local_method`, starting from `x`. Return a
`LocationValue`.
"""
function local_minimization(local_method::NLoptLocalMethod,
                            minimization_problem::MinimizationProblem, x)
    (; algorithm, xtol_abs, xtol_rel, maxeval, maxtime, ret_success) = local_method
    (; objective, lower_bounds, upper_bounds) = minimization_problem
    opt = NLopt.Opt(algorithm, length(x))
    opt.lower_bounds = lower_bounds
    opt.upper_bounds = upper_bounds

    # If a method `objective(x, grad)` exists, use it; otherwise assume objective is not
    # differentiable.
    opt.min_objective = applicable(objective, x, x) ? objective : nlopt_nondifferentiable_wrapper(objective)
    opt.xtol_abs = xtol_abs
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval
    opt.maxtime = maxtime
    optf, optx, ret = NLopt.optimize(opt, x)
    ret ∈ ret_success ? (value = optf, location = optx, ret = ret) : nothing
end

function nlopt_nondifferentiable_wrapper(fn)
    function f̃(x,grad)              # wrapper for NLopt
        @argcheck isempty(grad)     # ensure no derivatives are asked for
        return fn(x)
    end
    return f̃
end

end
