using MultistartOptimization
using Test
using NLopt: NLopt
using Distributed
using MultistartOptimization: local_minimization

include("test_functions.jl")

@testset "test function sanity checks" begin
    for F in TEST_FUNCTIONS
        @test F([minimum_location(F, 10)])[1] ≈ 0
    end
end
@testset "global optimization" begin
    for F in setdiff(TEST_FUNCTIONS, (RASTRIGIN, )) # Rastrigin disabled for now
        n = 10
        P = MinimizationProblem(F, lower_bounds(F, n), upper_bounds(F, n))
        local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
        multistart_method = TikTak(100)
        p = multistart_minimization(multistart_method, local_method, P)
        x₀ = minimum_location(F, n)
        @test p.location ≈ x₀ atol = 1e-5
        @test p.value ≈ F([x₀])[1] atol = 1e-10
    end
end
