using Documenter, ExaPF

makedocs(
    sitename = "ExaPF.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.MathJax2()
    ),
    strict = true,
    pages = [
        "Home" => "index.md",
        "AutoDiff" => "autodiff.md",
        "Linear Solver" => "linearsolver.md",
        "PowerSystem" => "powersystem.md",
        "Formulations" => "formulations.md",
        "Evaluators" => "evaluators.md",
    ]
)

