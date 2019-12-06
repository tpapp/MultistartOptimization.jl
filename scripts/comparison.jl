#####
##### simple comparison of various methods
#####

using MultistartOptimization
using NLopt
using Parameters

include("../test/test_functions.jl")

####
#### utility functions
####

function test(multistart_method, nlopt_method, f, N; options = ())
    n = 0
    function f̃(x)
        n += 1
        f(x)
    end
    F = MinimizationProblem(f̃, lower_bounds(f, N), upper_bounds(f, N))
    L = NLoptLocalMethod(nlopt_method; options...)
    @unpack location, value = multistart_minimization(multistart_method, L, F)
    (location = location, value = value, n = n)
end

function make_table(local_methods, functions, results; f_atol = 1e-3)
    codefence(str) = "`" * str * "`"
    table_body = map(x -> x.value ≤ f_atol ? string(x.n) : "**FAIL**", results)
    header_rows = [map(codefence ∘ string, local_methods)...]
    header_cols = [map(string ∘ Base.typename ∘ typeof, functions)...]
    table = ["" permutedims(header_cols);
             header_rows table_body]
end

function print_table(io, table)
    print_row(cells) = println(io, join(cells, " | "))
    for row in axes(table, 1)
        print_row(table[row, :])
        if row == firstindex(table, 1)
            print_row(fill("----", size(table, 2)))
        end
    end
end

###
### tests
###

METHODS = (NLopt.LN_BOBYQA, NLopt.LN_NELDERMEAD, NLopt.LN_NEWUOA_BOUND, NLopt.LN_SBPLX,
           NLopt.LN_COBYLA, NLopt.LN_PRAXIS)

mm = TikTak(100)

T10 = [test(mm, method, f, 10; options = (xtol_abs = 1e-8, ))
       for method in METHODS, f in TEST_FUNCTIONS]

table = make_table(METHODS, TEST_FUNCTIONS, T10)
print_table(stdout, table)
