# MultistartOptimization.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/MultistartOptimization.jl/workflows/CI/badge.svg)](https://github.com/tpapp/MultistartOptimization.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/MultistartOptimization.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/MultistartOptimization.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/MultistartOptimization.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/MultistartOptimization.jl/dev)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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

|  | ShiftedQuadratic | Griewank | LevyMontalvo2 | Rastrigin | Rosenbrock |
| ---- | ---- | ---- | ---- | ---- | ---- |
| `LN_BOBYQA` | 569 | 2633 | 4235 | **FAIL** | 10995 |
| `LN_NELDERMEAD` | 15750 | 17108 | 33088 | **FAIL** | 42785 |
| `LN_NEWUOA_BOUND` | 580 | 2088 | 2253 | **FAIL** | 13409 |
| `LN_SBPLX` | 12329 | 11806 | 11447 | **FAIL** | 7020038 |
| `LN_COBYLA` | 16943 | 37414 | 32792 | **FAIL** | 985676 |
| `LN_PRAXIS` | 1850 | 9886 | 8548 | **FAIL** | 15436 |
