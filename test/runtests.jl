using CUDA
using CUDA.CUSPARSE
using ExaPF
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs

import ExaPF: Parse, PowerSystem

Random.seed!(2713)

case = "case14.raw"
# case = "ACTIVSg70K.raw"

@testset "Powerflow residuals and Jacobian" begin
    # read data
    to = TimerOutputs.TimerOutput()
    datafile = joinpath(dirname(@__FILE__), case)
    data = Parse.parse_raw(datafile)
    BUS_B, BUS_AREA, BUS_VM, BUS_VA, BUS_NVHI, BUS_NVLO, BUS_EVHI,
    BUS_EVLO, BUS_TYPE = Parse.idx_bus()
    bus = data["BUS"]
    nbus = size(bus, 1)

    # obtain V0 from raw data
    V = Array{Complex{Float64}}(undef, nbus)
    T = Vector
    for i in 1:nbus
        V[i] = bus[i, BUS_VM]*exp(1im * pi/180 * bus[i, BUS_VA])
    end

    # form Y matrix
    Ybus, Yf_br, Yt_br, Yf_tr, Yt_tr = PowerSystem.makeYbus(data);

    Vm = abs.(V)
    Va = angle.(V)
    bus = data["BUS"]
    gen = data["GENERATOR"]
    load = data["LOAD"]
    nbus = size(bus, 1)
    ngen = size(gen, 1)
    nload = size(load, 1)

    ybus_re, ybus_im = ExaPF.Spmat{T}(Ybus)
    SBASE = data["CASE IDENTIFICATION"][1]
    Sbus = PowerSystem.assembleSbus(gen, load, SBASE, nbus)
    pbus = real(Sbus)
    qbus = imag(Sbus)

    ref, pv, pq = PowerSystem.bustypeindex(bus, gen)
    npv = size(pv, 1);
    npq = size(pq, 1);

    @testset "Computing residuals" begin
        F = zeros(Float64, npv + 2*npq)
        # First compute a reference value for resisual computed at V
        F♯ = ExaPF.residualFunction(V, Ybus, Sbus, pv, pq)
        # residual_polar! uses only binary types as this function is meant
        # to be deported on the GPU
        ExaPF.residualFunction_polar!(F, Vm, Va,
            ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
            ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
            pbus, qbus, pv, pq, nbus)
        @test F ≈ F♯
    end
    @testset "Computing Jacobian of residuals" begin
        F = zeros(Float64, npv + 2*npq)
        # Compute Jacobian at point V manually and use it as reference
        J = ExaPF.residualJacobian(V, Ybus, pv, pq)
        J♯ = copy(J)

        # Then, create a JacobianAD object
        coloring = ExaPF.matrix_colors(J)
        jacobianAD = ExaPF.AD.JacobianAD(J, coloring, F, Vm, Va, pv, pq)
        # and compute Jacobian with ForwardDiff
        ExaPF.AD.residualJacobianAD!(
            jacobianAD, ExaPF.residualFunction_polar!, Vm, Va,
            ybus_re, ybus_im, pbus, qbus, pv, pq, nbus, to)
        @test jacobianAD.J ≈ J♯
    end
end

@testset "Wrapping of iterative solvers" begin
    nblocks = 2
    n, m = 32, 32
    to = TimerOutputs.TimerOutput()

    # Add a diagonal term for conditionning
    A = randn(n, m) + 15I
    x♯ = randn(m)
    b = A * x♯

    # Be careful: all algorithms work with sparse matrix
    A = sparse(A)
    # Init iterators for KernelAbstractions
    if has_cuda_gpu()
        iterators = zip([CPU(), CUDADevice()],
                        [Array{Float64}, CuVector{Float64}],
                        [SparseMatrixCSC, CuSparseMatrixCSR])
    else
        iterators = zip([CPU()], [Array{Float64}], [SparseMatrixCSC])
    end
    @testset "Run iterative solvers on device $(device)" for (device, V, SM) in iterators
        As = SM(A)
        bs = convert(V, b)
        x0 = convert(V, zeros(m))
        xs♯ = convert(V, x♯)

        # TODO: currently Preconditioner takes as input only sparse matrix
        # defined on the main memory.
        precond = ExaPF.Precondition.Preconditioner(A, nblocks, device)

        # First test the custom implementation of BICGSTAB
        @testset "BICGSTAB" begin
            # Need to update preconditioner before resolution
            ExaPF.Precondition.update(As, precond, to)
            P = precond.P
            fill!(x0, 0.0)
            x_sol, n_iters = ExaPF.bicgstab(As, bs, P, x0, to)
            @test n_iters <= m
            @test x_sol ≈ xs♯ atol=1e-6
        end
        @testset "Interface for iterative algorithm ($algo)" for algo in ExaPF.list_solvers(device)
            fill!(x0, 0.0)
            n_iters = ExaPF.Iterative.ldiv!(x0, As, bs, algo, precond, to)
            @test n_iters <= m
            @test x0 ≈ xs♯ atol=1e-6
        end
    end
end

@testset "Powerflow solver" begin
    datafile = joinpath(dirname(@__FILE__), case)
    nblocks = 8
    # Create a network object:
    pf = ExaPF.PowerSystem.PowerNetwork(datafile)
    target = CPU()
    @testset "[CPU] Powerflow solver $precond" for precond in ExaPF.list_solvers(target)
        sol, has_conv, res = solve(pf, nblocks, precond, device=target)
        @test has_conv
        @test res < 1e-6
    end

    if has_cuda_gpu()
        target = CUDADevice()
        @testset "[GPU] Powerflow solver $precond" for precond in ExaPF.list_solvers(target)
            sol, conv, res = solve(pf, nblocks, precond, device=target)
            @test conv
            @test res < 1e-6
        end
    end
end

# # Not working yet. Will check whether Ipopt and reduced method match in objective
# @testset "rgm_3bus" begin
#    include("../scripts/rgm_3bus.jl")
#    @show red_cost = cfun(xk, uk, p)
#    include("../scripts/ipopt.jl")
#    @show ipopt_cost = cfun(xk, uk, p)
#    gap = abs(red_cost - ipopt_cost)
#    println("gap = abs(red_cost - ipopt_cost): $gap = abs($red_cost - $ipopt_cost)")
#    @test gap ≈ 0.0
# end

# @testset "rgm_3bus_ref" begin
#    include("../scripts/rgm_3bus_ref.jl")
#    @show red_cost = cfun(xk, uk, p)
#    include("../scripts/ipopt_ref.jl")
#    @show ipopt_cost = cfun(xk, uk, p)
#    gap = abs(red_cost - ipopt_cost)
#    println("gap = abs(red_cost - ipopt_cost): $gap = abs($red_cost - $ipopt_cost)")
#    @test gap ≈ 0.0
# end
