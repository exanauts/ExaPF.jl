using Pkg

Pkg.activate("$(@__DIR__)/docs")
Pkg.add(PackageSpec(path="$(@__DIR__)"))
Pkg.instantiate()
include("$(@__DIR__)/docs/make.jl")

