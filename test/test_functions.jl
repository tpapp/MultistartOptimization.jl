#####
##### Some multivariate test problems
#####

####
#### generic code
####

"""
A type for test functions with uniform bounds.

All subtypes (test functions) support:

1. being called with a vector of arbitrary length

2. `minimum_location`,

3. `lower_bounds`, `upper_bounds` (defined via `lu_bounds`),

4. having a global minimum *value* of `0`.

See [`TEST_FUNCTIONS`](@ref).
"""
abstract type UniformBounds end

lower_bounds(f::UniformBounds, n::Integer) = fill(Float64(lu_bounds(f)[1]), n)

upper_bounds(f::UniformBounds, n::Integer) = fill(Float64(lu_bounds(f)[2]), n)

####
#### shifted quadratic
####

"A test function with global minimum at [a, …]. For sanity checks."
struct ShiftedQuadratic{T <: Real} <: UniformBounds
    a::T
end

(f::ShiftedQuadratic)(x) = (a = f.a; sum(x -> abs2(x - a), x))

minimum_location(f::ShiftedQuadratic, n::Integer) = fill(f.a, n)

lu_bounds(f::ShiftedQuadratic) = (f.a - 50, f.a + 100)

const SHIFTED_QUADRATIC = ShiftedQuadratic(0.5)

####
#### Griewank
####

"""
The Griewank problem.

From Ali, Khompatraporn, and Zabinsy (2005, p 559).
"""
struct Griewank{T <: Real} <: UniformBounds
    a::T
end

function (f::Griewank)(x)
    sum(abs2, x) / f.a - prod(((i, x),) -> cos(x / √i), enumerate(x)) + 1
end

minimum_location(::Griewank, n::Integer) = zeros(n)

lu_bounds(::Griewank) = (-50, 100)

const GRIEWANK = Griewank(200.0)

####
#### Levy Montalvo 2.
####

"""
Levi and Montalvo 2 problem.

From Ali, Khompatraporn, and Zabinsy (2005, p 662).
"""
struct LevyMontalvo2 <: UniformBounds end

function (::LevyMontalvo2)(x)
    xn = last(x)
    0.1 * abs2(sinpi(3 * first(x))) + abs2(xn -  1) * (1 + abs2(sinpi(2 * xn))) +
        sum(@. abs2($(x[1:(end - 1)]) - 1) * (1 + abs2(sinpi(3 * $(x[2:end])))))
end

minimum_location(::LevyMontalvo2, n::Integer) = ones(n)

lu_bounds(::LevyMontalvo2) = (-10, 15)

const LEVY_MONTALVO2 = LevyMontalvo2()

####
#### Rastrigin
####

"""
Rastrigin problem.

From Ali, Khompatraporn, and Zabinsy (2005, p 665).
"""
struct Rastrigin <: UniformBounds end

function (::Rastrigin)(x)
    10 * length(x) + sum(@. abs2(x) - 10 * cospi(2 * x))
end

minimum_location(::Rastrigin, n::Integer) = zeros(n)

lu_bounds(::Rastrigin) = (-5.12, 5.12)

const RASTRIGIN = Rastrigin()

"""
Rosenbrock problem.

From Ali, Khompatraporn, and Zabinsy (2005, p 666).
"""
struct Rosenbrock <: UniformBounds end

function (::Rosenbrock)(x)
    x1 = x[1:(end - 1)]
    x2 = x[2:end]
    sum(@. 100 * abs2(x2 - abs2(x1)) + abs2(x1 - 1))
end

minimum_location(::Rosenbrock, n::Integer) = ones(n)

lu_bounds(::Rosenbrock) = (-30, 30)

####
#### Rosenbrock
####

const ROSENBROCK = Rosenbrock()

####
#### helper code
####

"A tuple of all test functions."
const TEST_FUNCTIONS = (SHIFTED_QUADRATIC, GRIEWANK, LEVY_MONTALVO2, RASTRIGIN, ROSENBROCK)
