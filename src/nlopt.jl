####
#### Local minimization with NLopt. Loaded on demand when `NLopt` is.
####

import NLopt

import MultistartOptimization: local_minimization

export NLoptLocalMethod

const NLopt_ret_success = Set([:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED,
                               :MAXEVAL_REACHED, :MAXTIME_REACHED])

Base.@kwdef struct NLoptLocalMethod{S}
    algorithm::NLopt.Algorithm
    xtol_abs::Float64 = 1e-8
    xtol_rel::Float64 = 1e-8
    maxeval::Int = 0
    maxtime::Float64 = 0.0
    "Return values which are considered as “success”."
    ret_success::S = NLopt_ret_success
end

"""
$(SIGNATURES)

A wrapper for algorithms supported by `NLopt`. Used to construct the corresponding
optimization problem. All return values in `ret_success` are considered valid (`ret` is also
kept in the result), all negative return values are considered invalid.

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
    @unpack algorithm, xtol_abs, xtol_rel, maxeval, maxtime, ret_success = local_method
    @unpack objective, lower_bounds, upper_bounds = minimization_problem
    opt = NLopt.Opt(algorithm, length(x))
    opt.lower_bounds = lower_bounds
    opt.upper_bounds = upper_bounds

    # if a method objective(x,grad) exists, use it. otherwise assume objective is not differentiable (which was the previous behavior)
    opt.min_objective = applicable(objective,x,x) ? objective : nloptwrapper(objective)
    opt.xtol_abs = xtol_abs
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval
    opt.maxtime = maxtime
    optf, optx, ret = NLopt.optimize(opt, x)
    ret ∈ ret_success ? (value = optf, location = optx, ret = ret) : nothing
end


function nloptwrapper(fn)  
    function f̃(x,grad)              # wrapper for NLopt
        @argcheck isempty(grad)     # ensure no derivatives are asked for
        return fn(x)
    end
    return f̃
end

