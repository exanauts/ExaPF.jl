using CUDA
using ExaPF
using KernelAbstractions
using Test

import ExaPF: PowerSystem

@testset "Powerflow solver" begin
    datafile = joinpath(dirname(@__FILE__), "..", "data", "case14.raw")
    pf = PowerSystem.PowerNetwork(datafile)
    # Parameters
    npartitions = 8
    tolerance = 1e-6
    if has_cuda_gpu()
        DEVICES = [CPU(), CUDADevice()]
    else
        DEVICES = [CPU()]
    end

    @testset "Deport computation on device $device" for device in DEVICES
        polar = PolarForm(pf, device)
        jac = ExaPF._state_jacobian(polar)
        precond = ExaPF.LinearSolvers.BlockJacobiPreconditioner(jac, npartitions, device)
        # Retrieve initial state of network
        x0 = ExaPF.initial(polar, State())
        uk = ExaPF.initial(polar, Control())

        @testset "Powerflow solver $(LinSolver)" for LinSolver in ExaPF.list_solvers(device)
            algo = LinSolver(precond)
            xk = copy(x0)
            nlp = ExaPF.ReducedSpaceEvaluator(polar, xk, uk;
                                              ε_tol=tolerance, linear_solver=algo)
            convergence = ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_NONE)
            @test convergence.has_converged
            @test convergence.norm_residuals < tolerance
            @test convergence.n_iterations == 2
        end
    end
    @testset "Deport only powerflow on device $device" for device in DEVICES
        polar = PolarForm(pf, CPU())
        jac = ExaPF._state_jacobian(polar)
        precond = ExaPF.LinearSolvers.BlockJacobiPreconditioner(jac, npartitions, CPU())
        # Retrieve initial state of network
        x0 = ExaPF.initial(polar, State())
        uk = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        @testset "Powerflow solver $(LinSolver)" for LinSolver in [ExaPF.LinearSolvers.DirectSolver]
            algo = LinSolver(precond)
            xk = copy(x0)
            nlp = ExaPF.ReducedSpaceEvaluator(polar, xk, uk, p;
                                              ε_tol=tolerance, linear_solver=algo)
            nlp_pf = ExaPF.ReducedSpaceEvaluator(nlp, device)
            convergence = ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_NONE, nlp_pf = nlp_pf)
            @test convergence.has_converged
            @test convergence.norm_residuals < tolerance
            @test convergence.n_iterations == 2
        end
    end
end
