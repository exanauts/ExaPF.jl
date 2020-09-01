
@testset "ReducedSpaceEvaluator" begin
    datafile = "test/data/case9.m"
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())
    x0 = ExaPF.initial(polar, State())
    u0 = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())

    nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p)

    @test ExaPF.n_variables(nlp) == length(u0)

    u = u0
    # Update nlp to stay on manifold
    ExaPF.update!(nlp, u)
    # Compute objective
    c = ExaPF.objective(nlp, u)
    @test isa(c, Real)

    g = similar(u)
    fill!(g, 0)
    ExaPF.gradient!(nlp, g, u)
end
