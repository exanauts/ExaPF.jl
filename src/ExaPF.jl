# Power flow module. The implementation is a modification of
# MATPOWER's code. We attach the original MATPOWER's license in makeYbus.m:
#
# MATPOWER
# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC)
# by Ray Zimmerman, PSERC Cornell
#
# Covered by the 3-clause BSD License.
module ExaPF

using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER
using ForwardDiff
using IterativeSolvers
using KernelAbstractions
using Krylov
using LinearAlgebra
using MathOptInterface
using Printf
using SparseArrays
using SparseDiffTools
using TimerOutputs

import Base: show, get

const MOI = MathOptInterface
const TIMER = TimerOutput()

const VERBOSE_LEVEL_HIGH = 3
const VERBOSE_LEVEL_MEDIUM = 2
const VERBOSE_LEVEL_LOW = 1
const VERBOSE_LEVEL_NONE = 0

include("utils.jl")
# Import submodules
include("ad.jl")
using .AD
include("indexes.jl")
using .IndexSet
include("LinearSolvers/LinearSolvers.jl")
using .LinearSolvers
include("parsers/parse_mat.jl")
using .ParseMAT
include("parsers/parse_psse.jl")
using .ParsePSSE
include("PowerSystem/PowerSystem.jl")
using .PowerSystem

const PS = PowerSystem

# Modeling
include("models/models.jl")
# Evaluators
include("Evaluators/Evaluators.jl")

end
