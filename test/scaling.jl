
@testset "ScalingEvaluator" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    pf = PowerSystem.PowerNetwork(datafile, 1)
    if has_cuda_gpu()
        ITERATORS = zip([CPU(), CUDADevice()], [Array, CuArray])
    else
        ITERATORS = zip([CPU()], [Array])
    end

    @testset "Test API on $device" for (device, M) in ITERATORS
        polar = PolarForm(pf, device)
        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
        nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p; constraints=constraints)
        ev = ExaPF.ScalingEvaluator(nlp, u0)

        # Test consistence
        n = ExaPF.n_variables(nlp)
        m = ExaPF.n_constraints(nlp)

        u = u0
        # Update nlp to stay on manifold
        ExaPF.update!(ev, u)
        # Compute objective
        c = ExaPF.objective(ev, u)
        @test isa(c, Real)
        # Compute gradient of objective
        g = similar(u)
        fill!(g, 0)
        ExaPF.gradient!(ev, g, u)
        function reduced_cost(u_)
            ExaPF.update!(ev, u_)
            return ExaPF.objective(ev, u_)
        end
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
        @test isapprox(grad_fd, g, rtol=1e-4)

        # Constraint
        ## Evaluation of the constraints
        cons = similar(u0, m)
        fill!(cons, 0)
        ExaPF.constraint!(nlp, cons, u)
        ## Evaluation of the Jacobian
        jac = similar(u0, m, n)
        fill!(jac, 0)
        ExaPF.jacobian!(ev, jac, u)
        ## Evaluation of the Jacobian transpose product
        v = similar(cons) ; fill!(v, 0)
        fill!(g, 0)
        ExaPF.jtprod!(ev, g, u, v)
        @test iszero(g)
        fill!(v, 1) ; fill!(g, 0)
        ExaPF.jtprod!(ev, g, u, v)
        @test isapprox(g, transpose(jac) * v)
    end
end
