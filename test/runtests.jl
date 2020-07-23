using Test
using CUDA

# This is a problem of the code right now. It can only set once per as
# this variable is used in macros to generate the code at compile time.
# This implies we cannot both test gpu and cpu code here.
target = "cpu"
@testset "Powerflow CPU" begin
    # Include code to run power flow equation
    include(joinpath(dirname(@__FILE__), "..", "examples", "pf.jl"))
    datafile = joinpath(dirname(@__FILE__), "case14.raw")
    # Direct solver
    sol, conv, res = pf(datafile)
    # test convergence is OK
    @test conv
    # test norm is minimized
    @test res < 1e-6

    nblocks = 8
    # Note: Reference BICGSTAB in IterativeSolvers
    for precond in ["gmres", "bicgstab_ref", "bicgstab"]
        sol, has_conv, res = pf(datafile, nblocks, precond)
        @test has_conv
        @test res < 1e-6
    end
end

## TODO: This throws warnings because the cpu version ran before.
if has_cuda_gpu()
    target = "cuda"
    @testset "Powerflow GPU" begin
        # Include code to run power flow equation
        include(joinpath(dirname(@__FILE__), "..", "examples", "pf.jl"))
        datafile = joinpath(dirname(@__FILE__), "case14.raw")
        # BICGSTAB
        sol, conv, res = pf(datafile, 2, "bicgstab")
        @test conv
        @test res < 1e-6
        # DIRECT
        sol, conv, res = pf(datafile)
        @test conv
        @test res < 1e-6
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
