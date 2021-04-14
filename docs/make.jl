using Pkg

Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "..")))
# when first running instantiate
Pkg.instantiate()

using Documenter, ExaPF

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
        "Manual" => [
            "AutoDiff" => "man/autodiff.md",
            "Benchmark" => "man/benchmark.md",
            "Linear Solver" => "man/linearsolver.md",
            "PowerSystem" => "man/powersystem.md",
            "Formulations" => "man/formulations.md",
            "Evaluators" => "man/evaluators.md",
        ],
        "Library" => [
            "AutoDiff" => "lib/autodiff.md",
            "Linear Solver" => "lib/linearsolver.md",
            "PowerSystem" => "lib/powersystem.md",
            "Formulations" => "lib/formulations.md",
            "Evaluators" => "lib/evaluators.md",
        ]
    ]
)

