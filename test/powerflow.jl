using CUDA
using ExaPF
using KernelAbstractions
using Test

import ExaPF: PowerSystem

@testset "Powerflow solver" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case14.raw")
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
        # Retrieve initial state of network
        x0 = ExaPF.initial(polar, State())
        uk = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        # BROKEN: Disable iterative solvers for the time being
        @testset "[CPU] Powerflow solver $precond" for precond in ExaPF.list_solvers(device)
            xk = copy(x0)
            nlp = ExaPF.ReducedSpaceEvaluator(polar, xk, uk, p;
                                                    ε_tol=tolerance, solver="$precond", npartitions=npartitions)
            convergence = @time ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_NONE)
            @test convergence.has_converged
            @test convergence.norm_residuals < tolerance
            @test convergence.n_iterations == 2
        end
    end
end
