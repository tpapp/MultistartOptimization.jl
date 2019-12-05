using MultistartOptimization
using Test
using NLopt: NLopt

using MultistartOptimization: local_minimization

"A test function with global minimum at [0.5, …]. For sanity checks."
f0(x) = sum(x -> abs2(x - 0.5), x)

P0 = MinimizationProblem(f0, zeros(5), ones(5))

local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)

local_minimization(local_method, P0, zeros(5))

t = TikTak(100)

p = global_minimization(t, local_method, P0)
@test p.location ≈ fill(0.5, 5)
@test p.value ≈ 0
