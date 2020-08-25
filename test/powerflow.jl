using CUDA
using ExaPF
using KernelAbstractions
using Test

@testset "Powerflow solver" begin
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
