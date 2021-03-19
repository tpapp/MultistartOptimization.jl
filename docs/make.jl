# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, MultistartOptimization

makedocs(
    modules = [MultistartOptimization],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Tamas K. Papp",
    sitename = "MultistartOptimization.jl",
    pages = Any["index.md"],
    strict = true,
    clean = true,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/tpapp/MultistartOptimization.jl.git",
    push_preview = true
)
