using MultistartOptimization
using Test
using NLopt: NLopt
using ForwardDiff

include("test_functions.jl")

@testset "test function sanity checks" begin
    for F in TEST_FUNCTIONS
        @test F(minimum_location(F, 10)) ≈ 0
    end
end

@testset "global optimization" begin
    for F in setdiff(TEST_FUNCTIONS, (RASTRIGIN, )) # Rastrigin disabled for now
        n = 10
        P = MinimizationProblem(F, lower_bounds(F, n), upper_bounds(F, n))
        local_method = NLopt_local_method(NLopt.LN_BOBYQA)
        multistart_method = TikTak(100)
        p = multistart_minimization(multistart_method, local_method, P)
        x₀ = minimum_location(F, n)
        @test p.location ≈ x₀ atol = 1e-5
        @test p.value ≈ F(x₀) atol = 1e-10
    end
end

@testset "custom local minimization and infeasibility" begin
    P = MinimizationProblem(x -> sum(abs2, x), [-1, -1], [1, 1])
    x0 = [0.0, 0.0]
    function _local_method(minimization_problem, x)
        if sum(abs2, x) < 0.25
            (location = x0, value = 0.0)
        else
            nothing
        end
    end
    p = multistart_minimization(TikTak(100), _local_method, P)
    @test p.location == x0
    @test p.value == 0.0
end


@testset "local gradient-based methods" begin
    function autodiff(fn)
        # adapted from here https://github.com/JuliaOpt/NLopt.jl/issues/128
        function f(x)
            return fn(x)
        end

        function f(x,∇f)
            if !(∇f == nothing) && (length(∇f) != 0)
                ForwardDiff.gradient!(∇f,fn,x)
            end

            fx = fn(x)
            return fx
        end
        return f
    end

    for F in setdiff(TEST_FUNCTIONS, (RASTRIGIN, )) # Rastrigin disabled for now
        n = 10
        P = MinimizationProblem(autodiff(F), lower_bounds(F, n), upper_bounds(F, n))
        local_method = NLopt_local_method(NLopt.LD_LBFGS)
        multistart_method = TikTak(100)
        p = multistart_minimization(multistart_method, local_method, P)
        x₀ = minimum_location(F, n)
        @test p.location ≈ x₀ atol = 1e-5
        @test p.value ≈ F(x₀) atol = 1e-10
    end

end

@testset "prepending points and sanity checks" begin
    N = 3
    lb = .-ones(N)
    ub = 2 .* ones(N)
    vz1 = -5.0
    z1 = lb .+ 1/√2 .* (ub .- lb) # special-cased, will be vz1
    g = function(x)
        # very narrow optimum near z1
        α = 1 - exp(-sum(abs2, 100 .* (x .- z1)))
        α * sum(abs2, x) + (1 - α) * vz1
    end
    P = MinimizationProblem(g, lb, ub)
    MM, LM = TikTak(100), NLopt_local_method(NLopt.LN_BOBYQA)
    r0 = multistart_minimization(MM, LM, P; use_threads = false)
    @test r0.value ≈ 0 atol = 1e-9         # sanity check
    r1 = multistart_minimization(MM, LM, P; use_threads = false, prepend_points = [z1])
    @test r1.value == vz1
    @test r1.location == z1
    @test_throws ArgumentError multistart_minimization(MM, LM, P; use_threads = false, prepend_points = [lb .- 2])
end

####
#### automated QA
####

import Aqua
Aqua.test_all(MultistartOptimization; stale_deps = false)
Aqua.test_stale_deps(MultistartOptimization, ignore = [:NLopt]) # conditional
import JET
JET.report_package("MultistartOptimization"; target_modules = [MultistartOptimization])
