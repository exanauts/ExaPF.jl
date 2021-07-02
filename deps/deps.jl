# This file pulls the dependencies using git branches for development

using Pkg
cusolverrf = PackageSpec(url="https://github.com/exanauts/BlockPowerFlow.jl.git", rev="master")
Pkg.add([cusolverrf])
