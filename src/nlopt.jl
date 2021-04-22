####
#### Local minimization with NLopt. Loaded on demand when `NLopt` is.
####

import NLopt

import MultistartOptimization: local_minimization

export NLoptLocalMethod

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
optimization problem. All positive return values are considered valid (`ret` is also kept in
the result), all negative return values are considered invalid.

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
    if (ret == NLopt.SUCCESS ||
        ret == NLopt.STOPVAL_REACHED ||
        ret == NLopt.FTOL_REACHED ||
        ret == NLopt.XTOL_REACHED ||
        ret == NLopt.MAXEVAL_REACHED ||
        ret == NLopt.MAXTIME_REACHED)
        (value = optf, location = optx, ret = ret)
    else
        nothing
    end
end
