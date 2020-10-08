@testset "PenaltyEvaluators" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    pf = PowerSystem.PowerNetwork(datafile, 1)

    polar = PolarForm(pf, CPU())
    x0 = ExaPF.initial(polar, State())
    u0 = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())

    constraints = Function[ExaPF.state_constraint]
    nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p; constraints=constraints)
    pen = ExaPF.PenaltyEvaluator(nlp)

    u = u0
    # Update nlp to stay on manifold
    ExaPF.update!(pen, u)
    # Compute objective
    c = ExaPF.objective(pen, u)
    @test isa(c, Real)
    # Compute gradient of objective
    g = similar(u)
    fill!(g, 0)
    ExaPF.gradient!(pen, g, u)
    #
    # Update penalty weigth
    ExaPF.update_penalty!(pen)
end
