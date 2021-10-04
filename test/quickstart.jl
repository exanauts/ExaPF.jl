using Test
using CUDA
using KernelAbstractions

using ExaPF
import ExaPF: AutoDiff
const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers

# Test quickstart guide in docs/src/quickstart.md
# If one test is broken, please update the documentation.

@testset "Documentation: quickstart" begin
    pglib_instance = "case1354.m"
    datafile = joinpath(dirname(@__FILE__), "..", "data", pglib_instance)

    # Short version
    polar = ExaPF.PolarForm(datafile, CPU())
    pf_algo = NewtonRaphson(; verbose=0, tol=1e-10)
    convergence = ExaPF.powerflow(polar, pf_algo)
    @test convergence.has_converged
    @test convergence.n_iterations == 5
    @test convergence.norm_residuals <= pf_algo.tol

    # Long version
    pf = PS.PowerNetwork(datafile)
    nbus = PS.get(pf, PS.NumberOfBuses())
    @test nbus == 1354

    pv_indexes = PS.get(pf, PS.PVIndexes())
    # Test only first PV index.
    @test pv_indexes[1] == 17

    # Build-up PolarForm object
    polar = ExaPF.PolarForm(pf, CPU())
    physical_state = get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, physical_state)
    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())

    linear_solver = LS.DirectSolver(jx.J)
    convergence = ExaPF.powerflow(
        polar, jx, physical_state, pf_algo;
        linear_solver=linear_solver
    )

    @test convergence.has_converged
    @test convergence.n_iterations == 5
    @test convergence.norm_residuals <= pf_algo.tol

    # Reinit buffer
    ExaPF.init_buffer!(polar, physical_state)
    npartitions = 8
    jac = jx.J
    precond = LS.BlockJacobiPreconditioner(jac, npartitions, CPU())
    iterative_linear_solver = ExaPF.KrylovBICGSTAB(jac; P=precond)
    @test isa(iterative_linear_solver, LS.AbstractIterativeLinearSolver)
    # Test default tolerance
    @test iterative_linear_solver.atol == 1e-10
    # Build powerflow algorithm
    pf_algo = NewtonRaphson(; verbose=0, tol=1e-7)

    convergence = ExaPF.powerflow(
        polar, jx, physical_state, pf_algo;
        linear_solver=iterative_linear_solver
    )

    @test convergence.has_converged
    @test convergence.n_iterations == 5
    @test convergence.norm_residuals <= pf_algo.tol

    if CUDA.has_cuda_gpu()
        polar_gpu = ExaPF.PolarForm(pf, CUDADevice())
        jx_gpu = AutoDiff.Jacobian(polar_gpu, ExaPF.power_balance, State())
        physical_state_gpu = get(polar_gpu, ExaPF.PhysicalState())
        ExaPF.init_buffer!(polar_gpu, physical_state_gpu)
        linear_solver = LS.DirectSolver(jx_gpu.J)
        convergence = ExaPF.powerflow(
            polar_gpu, jx_gpu, physical_state_gpu, pf_algo;
            linear_solver=linear_solver
        )

        @test convergence.has_converged
        @test convergence.n_iterations == 5
        @test convergence.norm_residuals <= pf_algo.tol

        npartitions = 8
        jac = jx_gpu.J # we need to take the Jacobian on the CPU for partitioning!
        precond = LS.BlockJacobiPreconditioner(jac, npartitions, CUDADevice())

        # Reinit buffer
        ExaPF.init_buffer!(polar_gpu, physical_state_gpu)

        linear_solver = ExaPF.KrylovBICGSTAB(jac; P=precond)
        pf_algo = NewtonRaphson(; verbose=0, tol=1e-7)
        convergence = ExaPF.powerflow(
            polar_gpu, jx_gpu, physical_state_gpu, pf_algo;
            linear_solver=linear_solver
        )

        @test convergence.has_converged
        @test convergence.n_iterations == 5
        @test convergence.norm_residuals <= pf_algo.tol
    end
end
