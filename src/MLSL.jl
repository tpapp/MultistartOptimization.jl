using DataStrucures

mutable struct Point

    iteration_no::Integer

    location::T
    value::S

    # store the local search results here
    minimized_res = nothing
end

mutable struct MLSL

    N::Integer
    γ::Float64
    r::Float64

    sample_points::RBTree{Point}() # tree for storing all the individual points

    minimas::RBTreeTree{Point}() # tree for storing the points on which local search has been performed

end


function MLSL(minimization_problem, local_search_algo, γ, r)

    # FIXME: Initialize all the appropriate struct variables
    mlsl = MLSL(N, )

    # TODO: Design the stopping criteria
    while(stopping_criteria == false):

        # Draw N points from SobolSeq and store them in MLSL.points
        S = sobol_starting_points(minimization_problem, quasirandom_N, false)

        # TODO: Use the Sobol.jl package directly and add the points to the tree
        # to save the unnecessary space allocated to S
        for point in S:
            push!(mlsl.sample_points, point)
        end


        # Rank them according to function values and select gamma*N from it...
        for _ in 1:trunc(Integer,mlsl.γ*N)
            # read minimum value
            pt = minimum_node(tree,sample_points.root)

            # check if we have to do a local search
            if(check_conditions_for_minimizing(mlsl, pt))
                continue
            end

            # do a local search
            pt.minimized_res = local_minimization(minimization_problem, pt.location)

            # store the local search results
            push!(mlsl.minimas,pt)

            # TODO: deallocate pt.minimized_res to `nothing` to save space, according to some condition
            # The garbage collector would still have to run to deallocate, so can I do better?



            # delete the point that we read (so that we don't read it again)
            delete!(sample_points,pt)
        end

    end
    return MLSL
end

# Utility Functions

# TODO: Add more parameters to utilize the flexibility of `optimize`
# local_minimization function for use with Optim.jl
function local_minimization(minimization_problem::MinimizationProblem, x0, algorithm = NelderMead, iterations = 1000)

    return optimize(minimization_problem, x0, algorithm(), iterations = 1000)

end

function check_conditions_for_minimizing(mlsl::MLSL, pt::Point)

    # check if our potential point lies very
    rt = mlsl.sample_points.root
    while (rt !== mlsl.minimas.nil)
        if sum(abs, pt.location .- rt.location) < r
            if pt.value > rt.value
                return true
            elseif
                rt = rt.leftChild
            end
        elseif pt.value > rt.value
            rt = rt.rightChild
        else
            rt = rt.leftChild
        end
        return false
    end

    rt = mlsl.minimas.root
    while (rt !== mlsl.minimas.nil)
        # TODO: Add some other criteria than just 'directly' comparing
        if pt.value == Optim.minimum(rt.minimized_res)
            return true
        elseif pt.value > Optim.minimum(rt.minimized_res)
            rt = rt.rightChild
        else
            rt = rt.leftChild
        end
        return false
    end
end


# The RedBlack Tree data structure from DataStructures.jl doesn't support
# overloading of primitive operations yet, so we need to overload them here.
function Point_comparator_lessthan(p::Point, q:: Point)
    if(p.minimized_res != nothing)
        return Optim.minimum(p.minimized_res) > Optim.minimum(q.minimized_res)
    else
        return p.value > q.value
    end
end

function Point_comparator_morethan(p::Point, q:: Point)
    if(p.minimized_res != nothing)
        return Optim.minimum(p.minimized_res) > Optim.minimum(q.minimized_res)
    else
        return p.value > q.value
    end
end

function Point_comparator_equalto(p::Point, q:: Point)
    if(p.minimized_res != nothing)
        return Optim.minimum(p.minimized_res) == Optim.minimum(q.minimized_res)
    else
        return p.value == q.value
    end
end

Base.:<(p::Point, q::Point) = Point_comparator_lessthan(p,q)
Base.:>(p::Point, q::Point) = Point_comparator_morethan(p,q)
Base.:==(p::Point, q::Point) = Point_comparator_equalto(p,q)
