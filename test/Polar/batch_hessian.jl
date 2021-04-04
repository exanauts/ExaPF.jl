using Printf
using CUDA
using CUDA.CUSPARSE
using FiniteDiff
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using ExaPF
import ExaPF: PowerSystem, AutoDiff

const PS = PowerSystem

@testset "Compute reduced Hessian on CPU" begin
    @testset "Case $case" for case in ["case9.m"]
        if has_cuda_gpu()
            ITERATORS = zip([CPU(), CUDADevice()], [Vector, CuVector])
        else
            ITERATORS = zip([CPU()], [Vector])
        end
        RTOL = 1e-6
        ATOL = 1e-6
        @testset "Device $device" for (device, T) in ITERATORS
            ##################################################
            # Initialization
            ##################################################
            datafile = joinpath(dirname(@__FILE__), "..", "..", "data", case)
            tolerance = 1e-8
            pf = PS.PowerNetwork(datafile)
            polar = PolarForm(pf, device)
            cpu_polar = PolarForm(pf, CPU())
            pv = pf.pv ; npv = length(pv)
            pq = pf.pq ; npq = length(pq)
            ref = pf.ref ; nref = length(ref)
            nbus = pf.nbus
            ngen = get(polar, PS.NumberOfGenerators())

            pv2gen = polar.indexing.index_pv_to_gen
            ref2gen = polar.indexing.index_ref_to_gen
            gen2bus = polar.indexing.index_generators
            cache = ExaPF.get(polar, ExaPF.PhysicalState())
            cpu_cache = ExaPF.get(cpu_polar, ExaPF.PhysicalState())
            ExaPF.init_buffer!(polar, cache)
            ExaPF.init_buffer!(cpu_polar, cpu_cache)

            xk = ExaPF.initial(polar, State())
            u = ExaPF.initial(polar, Control())
            nx = length(xk) ; nu = length(u)

            jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
            ju = AutoDiff.Jacobian(polar, ExaPF.power_balance, Control())
            ∂obj = ExaPF.AdjointStackObjective(polar)
            pbm = AutoDiff.TapeMemory(ExaPF.active_power_generation, ∂obj, nothing)

            cpu_jx = AutoDiff.Jacobian(cpu_polar, ExaPF.power_balance, State())
            cpu_ju = AutoDiff.Jacobian(cpu_polar, ExaPF.power_balance, Control())
            cpu_∂obj = ExaPF.AdjointStackObjective(cpu_polar)

            ##################################################
            # Step 1: computation of first-order adjoint
            ##################################################
            conv = powerflow(polar, jx, cache, NewtonRaphson())
            cpu_conv = powerflow(cpu_polar, cpu_jx, cpu_cache, NewtonRaphson())
            ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
            ExaPF.update!(cpu_polar, PS.Generators(), PS.ActivePower(), cpu_cache)
            @test conv.has_converged
            @test cpu_conv.has_converged
            # No need to recompute ∇gₓ
            ∇gₓ = jx.J
            ∇gᵤ = AutoDiff.jacobian!(polar, ju, cache)
            # test jacobian wrt x
            ∇gᵥ = AutoDiff.jacobian!(polar, jx, cache)
            @test isequal(∇gₓ, ∇gᵥ)

            # Fetch values found by Newton-Raphson algorithm
            vm = cache.vmag
            va = cache.vang
            pg = cache.pg
            # State & Control
            x = [va[pv] ; va[pq] ; vm[pq]]
            u = [vm[ref]; vm[pv]]
            # Test with Matpower's Jacobian
            V = vm .* exp.(im * va)
            Ybus = pf.Ybus
            Jₓ = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, V)
            @test isapprox(∇gₓ, Jₓ )
            # Hessian vector product
            ExaPF.gradient_objective!(polar, pbm, cache)
            ∇fₓ = ∂obj.∇fₓ
            ∇fᵤ = ∂obj.∇fᵤ
            λ  = -(Array(∇gₓ')) \ Array(∇fₓ)
            grad_adjoint = Array(∇fᵤ) + Array(∇gᵤ)' * λ

            ##################################################
            # Step 2: computation of Hessian of powerflow g
            ##################################################
            # Evaluate Hessian-vector product (full ∇²gₓₓ is a 3rd dimension tensor)
            ∇²gλ = ExaPF.matpower_hessian(cpu_polar, ExaPF.power_balance, cpu_cache, λ)
            ybus_re, ybus_im = ExaPF.Spmat{T{Int}, T{Float64}}(Ybus)
            pbus = T(real(pf.sbus))
            qbus = T(imag(pf.sbus))
            dev_pv = T(pf.pv)
            dev_pq = T(pf.pq)
            dev_ref = T(pf.ref)
            F = zeros(Float64, npv + 2*npq)
            nx = size(∇²gλ.xx, 1)
            nu = size(∇²gλ.uu, 1)

            tgt = rand(nx + nu)
            projp = zeros(nx + nu)
            single_H = AutoDiff.Hessian(polar, ExaPF.power_balance)
            @time AutoDiff.adj_hessian_prod!(polar, single_H, projp, cache, λ, tgt)

            nbatch = 32
            batch_H = ExaPF.batch_hessian(polar, ExaPF.power_balance, nbatch)

            btgt = rand(nbatch, nx + nu)
            bprojp = zeros(nbatch, nx + nu)
            H = [
                ∇²gλ.xx  ∇²gλ.xu' ;
                ∇²gλ.xu  ∇²gλ.uu
            ]
            @time ExaPF.batch_adj_hessian_prod!(polar, batch_H, bprojp, cache, λ, btgt)
            for i in 1:nbatch
                @test isapprox(bprojp[i, :], H * btgt[i, :])
            end
        end
    end
end

