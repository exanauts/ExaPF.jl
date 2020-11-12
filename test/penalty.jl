using CUDA
using CUDA.CUSPARSE
using ExaPF
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using FiniteDiff

import ExaPF: ParsePSSE, PowerSystem, IndexSet

@testset "PenaltyEvaluators" begin
    @testset "Inactive constraints" begin
        datafile = joinpath(dirname(@__FILE__), "..", "data", "case9.m")
        nlp = ExaPF.ReducedSpaceEvaluator(datafile)
        u0 = ExaPF.initial(nlp)
        # Build penalty evaluator
        pen = ExaPF.PenaltyEvaluator(nlp, u0)

        u = u0
        # Update nlp to stay on manifold
        ExaPF.update!(pen, u)
        # Compute objective
        c = ExaPF.objective(pen, u)
        c_ref = ExaPF.objective(nlp, u)
        @test isa(c, Real)
        # For case9.m all constraints are inactive and penalty are equal to zero
        @test iszero(pen.infeasibility)
        @test c == c_ref

        # Compute gradient of objective
        g = similar(u)
        g_ref = similar(u)
        fill!(g, 0)
        ExaPF.gradient!(pen, g, u)
        fill!(g_ref, 0)
        ExaPF.gradient!(nlp, g_ref, u)
        @test isequal(g_ref, g)
        # Update penalty weigth
        ExaPF.update_penalty!(pen)

        # Test reset
        ExaPF.reset!(pen)
    end
    @testset "Active constraints" begin
        datafile = joinpath(dirname(@__FILE__), "..", "data", "case57.m")
        nlp = ExaPF.ReducedSpaceEvaluator(datafile)
        u0 = ExaPF.initial(nlp)

        # Build penalty evaluator
        for scaling in [true, false]
            pen = ExaPF.PenaltyEvaluator(nlp, u0; scale=scaling)
            u = nlp.u_min
            # Update nlp to stay on manifold
            ExaPF.update!(pen, u)
            # Compute objective
            c = ExaPF.objective(pen, u)
            c_ref = ExaPF.objective(nlp, u)
            @test isa(c, Real)
            # For case57.m some constraints are active, so penalty are >= 0
            @test c > c_ref

            # Update penalty weigth with a large factor to have
            # a meaningful derivative check
            ExaPF.update_penalty!(pen, η=100.)
            # Compute gradient of objective
            g = similar(u)
            fill!(g, 0)
            ExaPF.gradient!(pen, g, u)
            # Compare with finite differences
            function reduced_cost(u_)
                ExaPF.update!(pen, u_)
                return ExaPF.objective(pen, u_)
            end
            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
            @test isapprox(grad_fd, g, rtol=1e-6)
        end
    end
end
