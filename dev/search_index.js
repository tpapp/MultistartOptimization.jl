var documenterSearchIndex = {"docs":
[{"location":"#MultistartOptimization","page":"MultistartOptimization","title":"MultistartOptimization","text":"","category":"section"},{"location":"#Introduction","page":"MultistartOptimization","title":"Introduction","text":"","category":"section"},{"location":"","page":"MultistartOptimization","title":"MultistartOptimization","text":"Multistart methods perform optimization from multiple starting points. This package implements multistart methods, relying on other packages for local methods. Wrappers are loaded on demand, or you can define them simply with a closure.","category":"page"},{"location":"#Example","page":"MultistartOptimization","title":"Example","text":"","category":"section"},{"location":"","page":"MultistartOptimization","title":"MultistartOptimization","text":"using MultistartOptimization, NLopt\n\nf(x) = sum(x -> abs2(x - 1), x)                     # objecive ∑(xᵢ-1)²\nP = MinimizationProblem(f, fill(-2, 4), fill(2, 4)) # search in [-2, 2]⁴\nlocal_method = NLoptLocalMethod(NLopt.LN_BOBYQA)    # loaded on demand with `using NLopt`\nmultistart_method = TikTak(100)\np = multistart_minimization(multistart_method, local_method, P)","category":"page"},{"location":"#Generic-API","page":"MultistartOptimization","title":"Generic API","text":"","category":"section"},{"location":"","page":"MultistartOptimization","title":"MultistartOptimization","text":"MinimizationProblem\nlocal_minimization\nmultistart_minimization","category":"page"},{"location":"#MultistartOptimization.MinimizationProblem","page":"MultistartOptimization","title":"MultistartOptimization.MinimizationProblem","text":"MinimizationProblem(#ctor-self#, objective, lower_bounds, upper_bounds)\n\n\nDefine a minimization problem.\n\nobjective is a function that maps ℝᴺ vectors to real numbers. It should accept all  AbstractVectors of length N.\nlower_bounds and upper_bounds define the bounds, and their lengths implicitly define the dimension N.\n\nThe fields of the result should be accessible with the names above.\n\n\n\n\n\n","category":"type"},{"location":"#MultistartOptimization.local_minimization","page":"MultistartOptimization","title":"MultistartOptimization.local_minimization","text":"local_minimization(minimization_problem, local_method, x)\n\nPerform a local minimization using local_method, starting from x.\n\nThe objective and the bounds are provided in minimization_problem, see MinimizationProblem. x should be an AbstractVector of conforming dimension.\n\nlocal_method can be a type for which a method is defined (recommended). However, it can also be a closure, in which case it will be called as local_method(minimization_problem, x).\n\nIn both cases, it should return nothing, or a value which has the following properties:\n\nlocation, an AbstractVector for the minimizer.\nvalue, for the value of the objective at location. Inf (or equivalent) should be used for infeasible regions.\n\nThe returned value may have other properties too, these are useful for convergence diagnostics, debugging information, etc, these depend on the local_method.\n\nReturning nothing is equivalent to value == Inf, but in some cases can work better for type inference as the method won't have to construct a counterfactual type.\n\n\n\n\n\nlocal_minimization(local_method, minimization_problem, x)\n\n\nSolve minimization_problem using local_method, starting from x. Return a LocationValue.\n\n\n\n\n\n","category":"function"},{"location":"#MultistartOptimization.multistart_minimization","page":"MultistartOptimization","title":"MultistartOptimization.multistart_minimization","text":"multistart_minimization(multistart_method, local_method, minimization_problem)\n\nMultistart minimization using the given multistart and local methods.\n\n\n\n\n\n","category":"function"},{"location":"#Multistart-methods","page":"MultistartOptimization","title":"Multistart methods","text":"","category":"section"},{"location":"","page":"MultistartOptimization","title":"MultistartOptimization","text":"TikTak","category":"page"},{"location":"#MultistartOptimization.TikTak","page":"MultistartOptimization","title":"MultistartOptimization.TikTak","text":"TikTak(quasirandom_N)\n\n\nThe “TikTak” multistart method, as described in Arnoud, Guvenen, and Kleineberg (2019).\n\nThis implements the multistart part, can be called with arbitrary local methods, see multistart_minimization.\n\nArguments\n\nquasirandom_N: the number of quasirandom points for the first pass (using a Sobol sequence).\n\nKeyword arguments\n\nkeep_ratio: the fraction of best quasirandom points which are kept\nθ_min and θ_max clamp the weight parameter, θ_pow determines the power it is raised to.\n\nThe defaults are from the paper cited above.\n\n\n\n\n\n","category":"type"},{"location":"#Local-methods","page":"MultistartOptimization","title":"Local methods","text":"","category":"section"},{"location":"","page":"MultistartOptimization","title":"MultistartOptimization","text":"Local methods are based on other optimization packages, and loaded on demand.","category":"page"},{"location":"#NLopt","page":"MultistartOptimization","title":"NLopt","text":"","category":"section"},{"location":"","page":"MultistartOptimization","title":"MultistartOptimization","text":"Available after NLopt is loaded, eg with using NLopt.","category":"page"},{"location":"","page":"MultistartOptimization","title":"MultistartOptimization","text":"NLoptLocalMethod","category":"page"},{"location":"#MultistartOptimization.NLoptLocalMethod","page":"MultistartOptimization","title":"MultistartOptimization.NLoptLocalMethod","text":"NLoptLocalMethod(algorithm)\n\n\nA wrapper for algorithms supported by NLopt. Used to construct the corresponding optimization problem. All return values in ret_success are considered valid (ret is also kept in the result), all negative return values are considered invalid.\n\nSee the NLopt documentation for the options. Defaults are changed slightly.\n\n\n\n\n\n","category":"type"}]
}
