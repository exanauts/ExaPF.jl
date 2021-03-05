# Check that interfaces of Evaluators are well implemented
@testset "API of evaluator $Evaluator" for Evaluator in [
    ExaPF.ReducedSpaceEvaluator,
    ExaPF.AugLagEvaluator,
    ExaPF.ProxALEvaluator,
    ExaPF.SlackEvaluator,
    ExaPF.FeasibilityEvaluator,
]
    datafile = joinpath(INSTANCES_DIR, "case9.m")
    # Default constructor: should take as input path to instance
    nlp = Evaluator(datafile)

    # Test printing
    println(devnull, nlp)

    # Test consistence
    n = ExaPF.n_variables(nlp)
    m = ExaPF.n_constraints(nlp)
    u = ExaPF.initial(nlp)

    u_min, u_max = ExaPF.bounds(nlp, ExaPF.Variables())
    g_min, g_max = ExaPF.bounds(nlp, ExaPF.Constraints())
    buffer = get(nlp, ExaPF.PhysicalState())

    @testset "Evaluator's API" begin
        @test n == length(u)
        @test length(u_min) == length(u_max) == n
        @test u_min <= u_max
        @test length(g_min) == length(g_max) == m
        if m > 0
            @test g_min <= g_max
        end

        @test isa(get(nlp, ExaPF.Constraints()), Array{Function})
        @test isa(get(nlp, State()), AbstractVector)
        @test isa(buffer, ExaPF.AbstractBuffer)
        @test ExaPF.constraints_type(nlp) in [:bound, :equality, :inequality]

        # setters
        nbus = get(nlp, PS.NumberOfBuses())
        loads = similar(u, nbus) ; fill!(loads, 1)
        ExaPF.setvalues!(nlp, PS.ActiveLoad(), loads)
        ExaPF.setvalues!(nlp, PS.ReactiveLoad(), loads)

        ExaPF.reset!(nlp)
    end

    @testset "Evaluator's callbacks" begin
        # 1/ update! function
        conv = ExaPF.update!(nlp, u)
        @test isa(conv, ExaPF.ConvergenceStatus)
        @test conv.has_converged

        # 2/ objective function
        c = ExaPF.objective(nlp, u)
        @test isa(c, Real)

        # 3/ gradient! function
        function reduced_cost(u_)
            ExaPF.update!(nlp, u_)
            return ExaPF.objective(nlp, u_)
        end
        g = similar(u) ; fill!(g, 0)
        ExaPF.gradient!(nlp, g, u)
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
        @test isapprox(g, grad_fd, rtol=1e-5)

        # 4/ Constraint
        ## Evaluation of the constraints
        if m > 0
            cons = similar(g_min) ; fill!(cons, 0.0)
            ExaPF.constraint!(nlp, cons, u)

            # Vector product
            jv = similar(u_min) ; fill!(jv, 0.0)
            v = similar(g_min) ; fill!(v, 1.0)
            ExaPF.jtprod!(nlp, jv, u, v)
            function reduced_cons(u_)
                ExaPF.update!(nlp, u_)
                ExaPF.constraint!(nlp, cons, u_)
                return dot(v, cons)
            end
            jv_fd = FiniteDiff.finite_difference_gradient(reduced_cons, u)

            @test isapprox(jv, jv_fd, rtol=1e-5)

            # Jacobian
            J =  ExaPF.jacobian(nlp, u)
            @test J' * v ≈ jv
        end
    end
end

