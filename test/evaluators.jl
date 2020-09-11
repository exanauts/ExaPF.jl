if has_cuda_gpu()
    DEVICES = [CPU(), CUDADevice()]
else
    DEVICES = [CPU()]
end

@testset "ReducedSpaceEvaluator $device" for device in DEVICES
    println("Device: $device")
    datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, device)
    x0 = ExaPF.initial(polar, State())
    u0 = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())

    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    print("Constructor\t")
    nlp = @time ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p; constraints=constraints)

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
    print("Update   \t")
    @time ExaPF.update!(nlp, u)
    # Compute objective
    print("Objective\t")
    c = @time ExaPF.objective(nlp, u)
    @test isa(c, Real)
    # Compute gradient of objective
    g = similar(u)
    fill!(g, 0)
    print("Gradient \t")
    @time ExaPF.gradient!(nlp, g, u)

    # Constraint
    ## Evaluation of the constraints
    g = zeros(m)
    print("Constrt \t")
    @time ExaPF.constraint!(nlp, g, u)
    ## Evaluation of the Jacobian
    print("Jacobian\t")
    J = @time ExaPF.jacobian!(nlp, u)
    # @info("j", J)
end
