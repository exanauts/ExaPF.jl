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
    @testset "Case $case" for case in ["case9.m", "case30.m"]
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
            ExaPF.gradient_objective!(polar, ∂obj, cache)
            ∇fₓ = ∂obj.∇fₓ
            ∇fᵤ = ∂obj.∇fᵤ
            λ  = -(Array(∇gₓ')) \ Array(∇fₓ)
            grad_adjoint = Array(∇fᵤ) + Array(∇gᵤ)' * λ

            ##################################################
            # Step 2: computation of Hessian of powerflow g
            ##################################################
            ## w.r.t. xx
            function jac_diff(x)
                vm_ = copy(vm)
                va_ = copy(va)
                va_[pv] = x[1:npv]
                va_[pq] = x[npv+1:npv+npq]
                vm_[pq] = x[npv+npq+1:end]
                V = vm_ .* exp.(im * va_)
                Jx = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, V)
                return Jx' * λ
            end

            # Evaluate Hessian-vector product (full ∇²gₓₓ is a 3rd dimension tensor)
            ∇²gλ = ExaPF.matpower_hessian(cpu_polar, ExaPF.power_balance, cpu_cache, λ)
            H_fd = FiniteDiff.finite_difference_jacobian(jac_diff, x)
            @test isapprox(∇²gλ.xx, H_fd, rtol=RTOL)
            ybus_re, ybus_im = ExaPF.Spmat{T{Int}, T{Float64}}(Ybus)
            pbus = T(real(pf.sbus))
            qbus = T(imag(pf.sbus))
            dev_pv = T(pf.pv)
            dev_pq = T(pf.pq)
            dev_ref = T(pf.ref)
            F = zeros(Float64, npv + 2*npq)
            nx = size(∇²gλ.xx, 1)
            nu = size(∇²gλ.uu, 1)

            # Hessian-vector product using forward over adjoint AD
            HessianAD = AutoDiff.Hessian(polar, ExaPF.power_balance)

            tgt = rand(nx + nu)
            projp = zeros(nx + nu)
            dev_tgt = T(tgt)
            dev_projp = T(projp)
            # set tangents only for x direction
            dev_λ = T(λ)
            tgt[nx+1:end] .= 0.0
            dev_tgt = T(tgt)
            dev_projxx = ExaPF.AutoDiff.tgt_adj_residual_hessian!(
                HessianAD, ExaPF.adj_residual_polar!, dev_λ, dev_tgt, vm, va,
                ybus_re, ybus_im, pbus, qbus, dev_pv, dev_pq, dev_ref, nbus)
            projxx = Array(dev_projxx)
            @test isapprox(projxx[1:nx], ∇²gλ.xx * tgt[1:nx])
            AutoDiff.adj_hessian_prod!(polar, HessianAD, dev_projp, cache, dev_λ, dev_tgt)
            projp = Array(dev_projp)
            @test isapprox(projp[1:nx], ∇²gλ.xx * tgt[1:nx])

            tgt = rand(nx + nu)
            # set tangents only for u direction
            tgt[1:nx] .= 0.0
            dev_tgt = T(tgt)
            dev_projuu = AutoDiff.tgt_adj_residual_hessian!(
                HessianAD, ExaPF.adj_residual_polar!, dev_λ, dev_tgt, vm, va,
                ybus_re, ybus_im, pbus, qbus, dev_pv, dev_pq, dev_ref, nbus)
            projuu = Array(dev_projuu)
            AutoDiff.adj_hessian_prod!(polar, HessianAD, dev_projp, cache, dev_λ, dev_tgt)
            projp = Array(dev_projp)
            # (we use absolute tolerance as Huu is equal to 0 for case9)
            @test isapprox(projuu[nx+1:end], ∇²gλ.uu * tgt[nx+1:end], atol=ATOL)
            @test isapprox(projp[nx+1:end], ∇²gλ.uu * tgt[nx+1:end], atol=ATOL)

            # check cross terms ux
            tgt = rand(nx + nu)
            tgt .= 1.0
            # Build full Hessian
            H = [
                ∇²gλ.xx ∇²gλ.xu';
                ∇²gλ.xu ∇²gλ.uu
            ]
            dev_tgt = T(tgt)
            dev_projxu = ExaPF.AutoDiff.tgt_adj_residual_hessian!(
                HessianAD, ExaPF.adj_residual_polar!, dev_λ, dev_tgt, vm, va,
                ybus_re, ybus_im, pbus, qbus, dev_pv, dev_pq, dev_ref, nbus)
            projxu = Array(dev_projxu)
            @test isapprox(projxu, H * tgt)
            AutoDiff.adj_hessian_prod!(polar, HessianAD, dev_projp, cache, dev_λ, dev_tgt)
            projp = Array(dev_projp)
            @test isapprox(projp, H * tgt)

            ## w.r.t. uu
            function jac_u_diff(u)
                vm_ = copy(vm)
                va_ = copy(va)
                vm_[ref] = u[1:nref]
                vm_[pv] = u[nref+1:end]
                V = vm_ .* exp.(im * va_)
                Ju = ExaPF.matpower_jacobian(polar, Control(), ExaPF.power_balance, V)
                return Ju' * λ
            end

            Hᵤᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_u_diff, u)

            ## w.r.t. xu
            function jac_xu_diff(x)
                vm_ = copy(vm)
                va_ = copy(va)
                va_[pv] = x[1:npv]
                va_[pq] = x[npv+1:npv+npq]
                vm_[pq] = x[npv+npq+1:end]
                V = vm_ .* exp.(im * va_)
                Ju = ExaPF.matpower_jacobian(polar, Control(), ExaPF.power_balance, V)[:, 1:nref+npv]
                return Ju' * λ
            end

            Hₓᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_xu_diff, x)
            @test isapprox(∇²gλ.xu[1:nref+npv, :], Hₓᵤ_fd, rtol=RTOL)

            ##################################################
            # Step 3: include constraints in Hessian
            ##################################################
            # h1 (state)      : xl <= x <= xu
            # h2 (by-product) : yl <= y <= yu
            # Test sequential evaluation of Hessian
            x = [va[pv] ; va[pq] ; vm[pq]]
            u = [vm[ref]; vm[pv]; pg[pv2gen]]

            μ = rand(ngen)
            ∂₂Q = ExaPF.matpower_hessian(cpu_polar, ExaPF.reactive_power_constraints, cpu_cache, μ)
            function jac_x(z)
                x_ = z[1:nx]
                u_ = z[1+nx:end]
                # Transfer control
                ExaPF.transfer!(polar, cache, u_)
                # Transfer state (manually)
                cache.vang[pv] .= x_[1:npv]
                cache.vang[pq] .= x_[npv+1:npv+npq]
                cache.vmag[pq] .= x_[npv+npq+1:end]
                ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
                J = ExaPF.jacobian(polar, ExaPF.reactive_power_constraints, cache)
                return [J.x J.u]' * μ
            end

            H_fd = FiniteDiff.finite_difference_jacobian(jac_x, [x; u])
            @test isapprox(∂₂Q.uu, H_fd[nx+1:end, nx+1:end], rtol=RTOL)
            @test isapprox(∂₂Q.xx, H_fd[1:nx, 1:nx], rtol=RTOL)
            @test isapprox(∂₂Q.xu, H_fd[nx+1:end, 1:nx], rtol=RTOL)

            # Test with AutoDiff.Hessian
            hess_reactive = AutoDiff.Hessian(polar, ExaPF.reactive_power_constraints)

            # XX
            tgt = rand(nx + nu)
            projp = zeros(nx + nu)
            tgt[nx+1:end] .= 0.0
            dev_tgt = T(tgt)
            dev_projp = T(projp)
            dev_μ = T(μ)
            AutoDiff.adj_hessian_prod!(polar, hess_reactive, dev_projp, cache, dev_μ, dev_tgt)
            projp = Array(dev_projp)
            @test isapprox(projp[1:nx], ∂₂Q.xx * tgt[1:nx])

            # UU
            tgt = rand(nx + nu)
            tgt[1:nx] .= 0.0
            dev_tgt = T(tgt)
            dev_projp .= 0.0
            AutoDiff.adj_hessian_prod!(polar, hess_reactive, dev_projp, cache, dev_μ, dev_tgt)
            projp = Array(dev_projp)
            @test isapprox(projp[1+nx:end], ∂₂Q.uu * tgt[1+nx:end])

            # XU
            tgt = rand(nx + nu)
            dev_tgt = T(tgt)
            dev_projp .= 0.0
            AutoDiff.adj_hessian_prod!(polar, hess_reactive, dev_projp, cache, dev_μ, dev_tgt)
            H = [
                ∂₂Q.xx ∂₂Q.xu' ;
                ∂₂Q.xu ∂₂Q.uu
            ]
            projp = Array(dev_projp)
            @test isapprox(projp, H * tgt)

            # Hessian w.r.t. Line-flow
            hess_lineflow = AutoDiff.Hessian(polar, ExaPF.flow_constraints)
            ncons = ExaPF.size_constraint(polar, ExaPF.flow_constraints)
            μ = rand(ncons)
            function flow_jac_x(z)
                x_ = z[1:nx]
                u_ = z[1+nx:end]
                # Transfer control
                ExaPF.transfer!(polar, cache, u_)
                # Transfer state (manually)
                cache.vang[pv] .= x_[1:npv]
                cache.vang[pq] .= x_[npv+1:npv+npq]
                cache.vmag[pq] .= x_[npv+npq+1:end]
                V = cache.vmag .* exp.(im .* cache.vang)
                Jx = ExaPF.matpower_jacobian(polar, State(), ExaPF.flow_constraints, V)
                Ju = ExaPF.matpower_jacobian(polar, Control(), ExaPF.flow_constraints, V)
                return [Jx Ju]' * μ
            end

            H_fd = FiniteDiff.finite_difference_jacobian(flow_jac_x, [x; u])
            tgt = rand(nx + nu)
            projp = zeros(nx + nu)
            dev_tgt = T(tgt)
            dev_projp = T(projp)
            dev_μ = T(μ)
            AutoDiff.adj_hessian_prod!(polar, hess_lineflow, dev_projp, cache, dev_μ, dev_tgt)
            projp = Array(dev_projp)
            @test isapprox(projp, H_fd * tgt, rtol=RTOL)

            # Hessian w.r.t. Active-power generation
            hess_pg = AutoDiff.Hessian(cpu_polar, ExaPF.active_power_constraints)
            μ = rand(1)
            # TODO: No GPU implementation so far
            AutoDiff.adj_hessian_prod!(cpu_polar, hess_pg, projp, cache, μ, tgt)
            ∂₂P = ExaPF.matpower_hessian(cpu_polar, ExaPF.active_power_constraints, cpu_cache, μ)
            H = [
                ∂₂P.xx ∂₂P.xu' ;
                ∂₂P.xu ∂₂P.uu
            ]
            @test isapprox(projp, (H * tgt))
        end
    end
end

