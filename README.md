# MultistartOptimization.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/tpapp/MultistartOptimization.jl.svg?branch=master)](https://travis-ci.com/tpapp/MultistartOptimization.jl)
[![codecov.io](http://codecov.io/github/tpapp/MultistartOptimization.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/MultistartOptimization.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/MultistartOptimization.jl/dev)

Experimenting with multistart optimization methods in Julia.

**WORK IN PROGRESS**. Expect API changes, but [SemVer 2](https://semver.org/) will of course be respected.

Documentation is very much WIP.

## How to use this package

1. Define a *minimization problem* with the objective, lower- and upper bounds,

2. pick a *local method* for each multistart point (currently methods in [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) are supported),

3. pick a *multistart method* (currently we have *TikTak* from *Arnoud, Guvenen, and Kleineberg (2019)*).

Example:

```julia
using MultistartOptimization, NLopt
P = MinimizationProblem(x -> sum(abs2, x), -ones(10), ones(10))
local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
multistart_method = TikTak(100)
p = multistart_minimization(multistart_method, local_method, P)
p.location, p.value
```

## Some benchmarks

Number of function evaluations for

- `TikTak` with 100 Sobol initial points,
- dimension `10`,
- local search terminating with absolute tolerance `1e-8` in the position

 | ShiftedQuadratic | Griewank | LevyMontalvo2 | Rastrigin | Rosenbrock
---- | ---- | ---- | ---- | ---- | ----
`LN_BOBYQA` | 572 | 2681 | 4236 | **FAIL** | 11321
`LN_NELDERMEAD` | 15678 | 17256 | 33120 | **FAIL** | 50331
`LN_NEWUOA_BOUND` | 582 | 2064 | 2183 | **FAIL** | 17817
`LN_SBPLX` | 12319 | 11795 | 11280 | **FAIL** | 6788573
`LN_COBYLA` | 16921 | **FAIL** | 32806 | **FAIL** | **FAIL**
`LN_PRAXIS` | 1830 | 8975 | 8239 | **FAIL** | 15438
