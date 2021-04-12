using CUDA
using CUDAKernels
using ExaPF
using KernelAbstractions
using Test

import ExaPF: PowerSystem
import ExaPF: LinearSolvers
const LS = LinearSolvers

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
        precond = ExaPF.build_preconditioner(polar; nblocks=npartitions)
        # Retrieve initial state of network
        uk = ExaPF.initial(polar, Control())

        @testset "Powerflow solver $(LinSolver)" for LinSolver in ExaPF.list_solvers(device)
            (LinSolver == LS.DirectSolver) && continue
            algo = LinSolver(precond)
            nlp = ExaPF.ReducedSpaceEvaluator(
                polar;
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
