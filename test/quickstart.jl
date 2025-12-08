using Test
using KernelAbstractions
using KrylovPreconditioners

using ExaPF
import ExaPF: AutoDiff
const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers

# Test quickstart guide in docs/src/quickstart.md
# If one test is broken, please update the documentation.

@testset "Documentation: quickstart" begin
    case = "case1354.m"
    # Short version
    polar = ExaPF.load_polar(case, CPU())
    # Initial values
    stack = ExaPF.NetworkStack(polar)
    convergence = run_pf(polar, stack; rtol=1e-10)
    @test convergence.has_converged
    @test convergence.n_iterations <= 6
    @test convergence.norm_residuals <= 1e-10

    # Long version
    pf = PS.load_case(case)
    nbus = PS.get(pf, PS.NumberOfBuses())
    @test nbus == 1354

    pv_indexes = pf.pv
    # Test only first PV index.
    @test pv_indexes[1] == 17

    # Build-up PolarForm object
    polar = ExaPF.PolarForm(pf, CPU())
    stack = ExaPF.NetworkStack(polar)
    basis = ExaPF.PolarBasis(polar)
    # Powerflow function
    pflow = ExaPF.PowerFlowBalance(polar) ∘ basis
    # AD for Jacobian
    jx = ExaPF.Jacobian(polar, pflow, State())
    # Linear solver
    linear_solver = LS.DirectSolver(jx.J)
    # Powerflow solver
    pf_solver = NewtonRaphson(tol=1e-10)

    convergence = ExaPF.nlsolve!(
        pf_solver, jx, stack; linear_solver=linear_solver,
    )

    @test convergence.has_converged
    @test convergence.n_iterations <= 6
    @test convergence.norm_residuals <= pf_solver.tol

    # Reinit buffer
    ExaPF.init!(polar, stack)
    npartitions = 8
    jac = jx.J
    precond = BlockJacobiPreconditioner(jac, npartitions, CPU())
    iterative_linear_solver = ExaPF.Bicgstab(jac; P=precond)
    @test isa(iterative_linear_solver, LS.AbstractIterativeLinearSolver)
    # Test default tolerance
    @test iterative_linear_solver.atol == 1e-10
    # Build powerflow algorithm
    pf_algo = NewtonRaphson(; verbose=0, tol=1e-7)

    convergence = ExaPF.nlsolve!(
        pf_solver, jx, stack; linear_solver=iterative_linear_solver,
    )

    @test convergence.has_converged
    @test convergence.n_iterations <= 6
    @test convergence.norm_residuals <= pf_algo.tol

    if test_cuda
        println("This runs on CUDA...")
        polar_gpu = ExaPF.PolarForm(pf, CUDABackend())
        stack_gpu = ExaPF.NetworkStack(polar_gpu)

        basis_gpu = ExaPF.PolarBasis(polar_gpu)
        pflow_gpu = ExaPF.PowerFlowBalance(polar_gpu) ∘ basis_gpu
        jx_gpu = ExaPF.Jacobian(polar_gpu, pflow_gpu, State())

        linear_solver = LS.DirectSolver(jx_gpu.J)

        convergence = ExaPF.nlsolve!(
            pf_solver, jx_gpu, stack_gpu; linear_solver=linear_solver,
        )

        @test convergence.has_converged
        @test convergence.n_iterations <= 6
        @test convergence.norm_residuals <= pf_solver.tol

        npartitions = 8
        jac = jx_gpu.J # we need to take the Jacobian on the CPU for partitioning!
        precond = BlockJacobiPreconditioner(jac, npartitions, CUDABackend())

        # Reinit buffer
        ExaPF.init!(polar_gpu, stack_gpu)

        iterative_linear_solver = ExaPF.Bicgstab(jac; P=precond)

        convergence = ExaPF.nlsolve!(
            pf_solver, jx_gpu, stack_gpu; linear_solver=iterative_linear_solver,
        )

        @test convergence.has_converged
        # Evalutates to 5 or 6 on GPU depending on numerical differences
        @test convergence.n_iterations <= 6
        @test convergence.norm_residuals <= pf_solver.tol
    end
end
