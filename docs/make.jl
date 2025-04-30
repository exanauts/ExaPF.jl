using Pkg

Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "..")))
# # when first running instantiate
Pkg.instantiate()

using Documenter
using ExaPF
using Documenter.Remotes

makedocs(
    sitename = "ExaPF.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX(),
    ),
    modules = [ExaPF],
    repo = Remotes.GitHub("exanauts", "ExaPF.jl"),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Quick start" => "quickstart.md",
        "Tutorials" => [
            "Power flow: direct solver" => "tutorials/direct_solver.md",
            "Power flow: batch evaluation" => "tutorials/batch_evaluation.md",
        ],
        "Manual" => [
            "PowerSystem" => "man/powersystem.md",
            "Polar formulation" => "man/formulations.md",
            "AutoDiff" => "man/autodiff.md",
            "Linear Solvers" => "man/linearsolver.md",
            "Benchmark" => "man/benchmark.md",
        ],
        "Library" => [
            "PowerSystem" => "lib/powersystem.md",
            "Polar formulation" => "lib/formulations.md",
            "AutoDiff" => "lib/autodiff.md",
            "Linear Solvers" => "lib/linearsolver.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/exanauts/ExaPF.jl.git",
    target = "build",
    devbranch = "main",
    devurl = "dev",
    push_preview = true,
)
