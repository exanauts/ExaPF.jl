using CUDA
using ExaPF
using KernelAbstractions
using Test

import ExaPF: PowerSystem

@testset "Powerflow solver" begin
    datafile = joinpath(INSTANCES_DIR, "case14.raw")
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
        jac = ExaPF.residual_jacobian(State(), polar)
        precond = ExaPF.LinearSolvers.BlockJacobiPreconditioner(jac, npartitions, device)
        # Retrieve initial state of network
        x0 = ExaPF.initial(polar, State())
        uk = ExaPF.initial(polar, Control())

        @testset "Powerflow solver $(LinSolver)" for LinSolver in ExaPF.list_solvers(device)
            algo = LinSolver(precond)
            xk = copy(x0)
            nlp = ExaPF.ReducedSpaceEvaluator(
                polar, xk, uk;
                powerflow_solver=NewtonRaphson(tol=tolerance, verbose=0),
                linear_solver=algo
            )
            convergence = ExaPF.update!(nlp, uk)
            @test convergence.has_converged
            @test convergence.norm_residuals < tolerance
            @test convergence.n_iterations == 2
        end
    end
end