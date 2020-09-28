using Documenter, ExaPF

makedocs(
    sitename = "ExaPF.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    strict = true,
    pages = [
        "Home" => "index.md",
        "Quick start" => "quickstart.md",
        "Manual" => [
            "AutoDiff" => "man/autodiff.md",
            "Linear Solver" => "man/linearsolver.md",
            "PowerSystem" => "man/powersystem.md",
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

