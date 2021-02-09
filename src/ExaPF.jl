# Power flow module. The implementation is a modification of
# MATPOWER's code. We attach the original MATPOWER's license in makeYbus.m:
#
# MATPOWER
# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC)
# by Ray Zimmerman, PSERC Cornell
#
# Covered by the 3-clause BSD License.
module ExaPF

# Standard library
using Printf
using LinearAlgebra
using SparseArrays

import CUDA
import CUDA.CUBLAS
import CUDA.CUSPARSE
import CUDA.CUSOLVER

import KernelAbstractions
const KA = KernelAbstractions
import MathOptInterface
const MOI = MathOptInterface
using TimerOutputs: @timeit, TimerOutput
import Zygote

import Base: show, get

const VERBOSE_LEVEL_HIGH = 3
const VERBOSE_LEVEL_MEDIUM = 2
const VERBOSE_LEVEL_LOW = 1
const VERBOSE_LEVEL_NONE = 0
const TIMER = TimerOutput()

include("utils.jl")
include("architectures.jl")
# Templates
include("models.jl")

# Import submodules
include("autodiff.jl")
using .AutoDiff
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
include("caches.jl")
# Polar formulation:
include("polar/polar.jl")

# Evaluators
include("Evaluators/Evaluators.jl")

end
