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
    @testset "Case $case" for case in ["case1354.m"]
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

            Ybus = pf.Ybus

            ##################################################
            # Computation of Hessian of powerflow g
            ##################################################
            λ = ones(nx)
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

            λ = λ |> T
            tgt = rand(nx + nu) |> T
            projp = zeros(nx + nu) |> T
            single_H = AutoDiff.Hessian(polar, ExaPF.power_balance)
            AutoDiff.adj_hessian_prod!(polar, single_H, projp, cache, λ, tgt)

            nbatch = 64
            batch_H = ExaPF.batch_hessian(polar, ExaPF.power_balance, nbatch)
            ExaPF.update!(polar, batch_H, cache)

            MT = isa(device, CUDADevice) ? CuMatrix : Matrix

            btgt = rand(nx + nu, nbatch) |> MT
            bprojp = zeros(nx + nu, nbatch) |> MT
            H = [
                ∇²gλ.xx  ∇²gλ.xu' ;
                ∇²gλ.xu  ∇²gλ.uu
            ]
            ExaPF.batch_adj_hessian_prod!(polar, batch_H, bprojp, cache, λ, btgt)
            if !isa(device, CUDADevice)
                for i in 1:nbatch
                    @test isapprox(bprojp[:, i], H * btgt[:, i])
                end
            end
        end
    end
end

