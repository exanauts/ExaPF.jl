using Pkg

Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "..")))
# # when first running instantiate
Pkg.instantiate()

using Documenter
using ExaPF

makedocs(
    sitename = "ExaPF.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [ExaPF],
    repo = "https://github.com/exanauts/ExaPF.jl/blob/{commit}{path}#{line}",
    strict = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Quick start" => "quickstart.md",
        "Tutorials" => [
            "Power flow: direct solver" => "tutorials/direct_solver.md",
        ],
        "Manual" => [
            "Polar formulation" => "man/formulations.md",
            "PowerSystem" => "man/powersystem.md",
            "AutoDiff" => "man/autodiff.md",
            "Linear Solvers" => "man/linearsolver.md",
            "Benchmark" => "man/benchmark.md",
        ],
        "Library" => [
            "Polar formulation" => "lib/formulations.md",
            "PowerSystem" => "lib/powersystem.md",
            "AutoDiff" => "lib/autodiff.md",
            "Linear Solvers" => "lib/linearsolver.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/exanauts/ExaPF.jl.git",
    target = "build",
    devbranch = "master",
    devurl = "dev",
    push_preview = true,
)
