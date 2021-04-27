# MultistartOptimization

## Introduction

Multistart methods perform optimization from multiple starting points. This package implements multistart methods, relying on other packages for local methods. Wrappers are loaded on demand, or you can define them simply with a closure.

### Example

```@example
using MultistartOptimization, NLopt

f(x) = sum(x -> abs2(x - 1), x)                     # objecive ∑(xᵢ-1)²
P = MinimizationProblem(f, fill(-2, 4), fill(2, 4)) # search in [-2, 2]⁴
local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)    # loaded on demand with `using NLopt`
multistart_method = TikTak(100)
p = multistart_minimization(multistart_method, local_method, P)
```

## Generic API

```@docs
MinimizationProblem
local_minimization
multistart_minimization
```

## Multistart methods

```@docs
TikTak
```

## Local methods

Local methods are based on other optimization packages, and loaded on demand.

### NLopt

Available after `NLopt` is loaded, eg with `using NLopt`.

```@docs
NLoptLocalMethod
```
