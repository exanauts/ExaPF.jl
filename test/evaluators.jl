@testset "ReducedSpaceEvaluators" begin
    if has_cuda_gpu()
        ITERATORS = zip([CPU(), CUDADevice()], [Array, CuArray])
    else
        ITERATORS = zip([CPU()], [Array])
    end
    datafile = joinpath(dirname(@__FILE__), "data", "case300.m")
    pf = PowerSystem.PowerNetwork(datafile, 1)

    @testset "Test API on $device" for (device, M) in ITERATORS
        println("Device: $device")
        polar = PolarForm(pf, device)
        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
        print("Constructor\t")
        nlp = @time ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p; constraints=constraints)

        # Test evaluator is well instantiated on target device
        TNLP = typeof(nlp)
        for (fn, ft) in zip(fieldnames(TNLP), fieldtypes(TNLP))
            if ft <: AbstractArray{Float64, P} where P
                @test isa(getfield(nlp, fn), M)
            end
        end

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
        CUDA.@time ExaPF.update!(nlp, u)
        # Compute objective
        print("Objective\t")
        c = CUDA.@time ExaPF.objective(nlp, u)
        @test isa(c, Real)
        # Compute gradient of objective
        g = similar(u)
        fill!(g, 0)
        print("Gradient \t")
        CUDA.@time ExaPF.gradient!(nlp, g, u)

        # Constraint
        ## Evaluation of the constraints
        cons = similar(nlp.g_min)
        fill!(cons, 0)
        print("Constrt \t")
        CUDA.@time ExaPF.constraint!(nlp, cons, u)
        ## Evaluation of the Jacobian
        # print("Jacobian\t")
        jac = M{Float64, 2}(undef, m, n)
        fill!(jac, 0)
        # @time ExaPF.jacobian!(nlp, jac, u)
        # @info("j", J)
    end

    # Test correctness of the reduced gradient (currently only on CPU)
    CASES = ["case9.m", "case30.m"]
    @testset "Evaluation of reduced gradient on $case" for case in CASES
        datafile = joinpath(dirname(@__FILE__), "data", case)
        pf = PowerSystem.PowerNetwork(datafile, 1)
        polar = PolarForm(pf, CPU())
        x0 = ExaPF.initial(polar, State())
        u = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
        nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u, p; constraints=constraints)
        ExaPF.update!(nlp, u)
        ∇fᵣ = similar(u)
        fill!(∇fᵣ, 0)
        ExaPF.gradient!(nlp, ∇fᵣ, u)

        # Compare with finite differences
        function reduced_cost(u_)
            ExaPF.update!(nlp, u_)
            return ExaPF.objective(nlp, u_)
        end
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
        @test isapprox(grad_fd, ∇fᵣ, rtol=1e-4)
    end
end
