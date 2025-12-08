module ExaPF

# Standard library
using Printf
using LinearAlgebra
using SparseArrays
import ForwardDiff
import SparseMatrixColorings
using KernelAbstractions
using GPUArraysCore
const KA = KernelAbstractions

import Base: show, get

export run_pf
export State, Control, AllVariables, PolarForm, BlockPolarForm, PolarFormRecourse

# Export KernelAbstractions backends
export CPU

include("templates.jl")
include("utils.jl")

include("autodiff.jl")
using .AutoDiff
include("LinearSolvers/LinearSolvers.jl")
using .LinearSolvers
include("PowerSystem/PowerSystem.jl")
using .PowerSystem

const PS = PowerSystem
const LS = LinearSolvers

# Polar formulation
include("Polar/polar.jl")

function run_pf(datafile::String, backend::KA.Backend, )
    polar = ExaPF.PolarForm(datafile, backend)
    stack = ExaPF.NetworkStack(polar)
    conv = ExaPF.run_pf(polar, stack)
    return conv
end
end
