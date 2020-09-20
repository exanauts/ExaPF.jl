using Documenter, ExaPF

makedocs(
    sitename = "ExaPF",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.MathJax()
    ),
    strict = true,
    pages = [
        "Introduction" => "index.md",
        "PowerSystem" => "powersystem.md",
        "Formulations" => "formulations.md",
        "Evaluators" => "evaluators.md",
    ]
)

