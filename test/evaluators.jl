
@testset "ReducedSpaceEvaluator" begin
    datafile = "test/data/case9.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())
    x0 = ExaPF.initial(polar, State())
    u0 = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())

    constraints = [ExaPF.state_constraint, ExaPF.power_constraints]
    nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p; constraints=constraints)

    # Test consistence
    n = ExaPF.n_variables(nlp)
    m = ExaPF.n_constraints(nlp)
    @test n == length(u0)
    @test isless(nlp.u_min, nlp.u_max)
    @test isless(nlp.x_min, nlp.x_max)
    @test isless(nlp.g_min, nlp.g_max)
    @test length(nlp.g_min) == m

    u = u0
    # Update nlp to stay on manifold
    ExaPF.update!(nlp, u)
    # Compute objective
    c = ExaPF.objective(nlp, u)
    @test isa(c, Real)
    # Compute gradient of objective
    g = similar(u)
    fill!(g, 0)
    ExaPF.gradient!(nlp, g, u)

    # Constraint
    ## Evaluation of the constraints
    g = zeros(m)
    ExaPF.constraint!(nlp, g, u)
end
