using CUDA
using ExaPF
using KernelAbstractions
using Test

import ExaPF: PowerSystem

@testset "Powerflow solver DEPRECATED API" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case14.raw")
    # Parameters
    nblocks = 8
    tolerance = 1e-6

    # Create a powersystem object:
    pf = ExaPF.PowerSystem.PowerNetwork(datafile)

    # Retrieve initial state of network
    pbus = real.(pf.sbus)
    qbus = imag.(pf.sbus)
    vmag = abs.(pf.vbus)
    vang = angle.(pf.vbus)

    x = ExaPF.PowerSystem.get_x(pf, vmag, vang, pbus, qbus)
    u = ExaPF.PowerSystem.get_u(pf, vmag, vang, pbus, qbus)
    p = ExaPF.PowerSystem.get_p(pf, vmag, vang, pbus, qbus)

    target = CPU()
    # BROKEN: Disable iterative solvers for the time being
    # @testset "[CPU] Powerflow solver $precond" for precond in ExaPF.list_solvers(target)
    @testset "[CPU] Powerflow solver $precond" for precond in ["default"]
        sol, g, Jx, Ju, convergence = solve(pf, x, u, p;
                                            npartitions=nblocks,
                                            solver=precond,
                                            device=target,
                                            tol=tolerance)
        @test convergence.has_converged
        @test convergence.norm_residuals < tolerance
        @test convergence.n_iterations == 2
    end

    if has_cuda_gpu()
        target = CUDADevice()
        @testset "[GPU] Powerflow solver $precond" for precond in ["default"]
            sol, g, Jx, Ju, convergence = solve(pf, x, u, p;
                                                npartitions=nblocks,
                                                solver=precond,
                                                device=target,
                                                tol=tolerance)
            @test convergence.has_converged
            @test convergence.norm_residuals < tolerance
            @test convergence.n_iterations == 2
        end
    end
end

# @testset "Powerflow solver" begin
function main()
    datafile = joinpath(dirname(@__FILE__), "data", "case14.raw")
    # Parameters
    npartitions = 8
    tolerance = 1e-6

    # Create a powersystem object:
    @testset "[CPU] Powerflow solver $precond" for precond in ["default"]
        pf = PowerSystem.PowerNetwork(datafile)
        polar = PolarForm(pf, CPU(); nocost=true)

        # Retrieve initial state of network
        xk = ExaPF.initial(polar, State())
        uk = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        # BROKEN: Disable iterative solvers for the time being
        # @testset "[CPU] Powerflow solver $precond" for precond in ExaPF.list_solvers(target)
        nlp = @time ExaPF.ReducedSpaceEvaluator(polar, xk, uk, p; 
                                                ε_tol = tolerance, solver="$precond", npartitions=npartitions)
        convergence = ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_NONE)
        @test convergence.has_converged
        @test convergence.norm_residuals < tolerance
        @test convergence.n_iterations == 2
    end

    if has_cuda_gpu()
        @testset "[GPU] Powerflow solver $precond" for precond in ["default"]
            pf = PowerSystem.PowerNetwork(datafile)
            polar = PolarForm(pf, CUDADevice(); nocost=true)

            # Retrieve initial state of network
            xk = ExaPF.initial(polar, State())
            uk = ExaPF.initial(polar, Control())
            p = ExaPF.initial(polar, Parameters())

            nlp = @time ExaPF.ReducedSpaceEvaluator(polar, xk, uk, p; 
                                                    ε_tol = tolerance, solver="$precond", npartitions=npartitions)
            convergence = ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_NONE)
            @test convergence.has_converged
            @test convergence.norm_residuals < tolerance
            @test convergence.n_iterations == 2
        end
    end
end