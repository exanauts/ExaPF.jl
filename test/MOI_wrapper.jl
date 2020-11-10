
using Ipopt
using MathOptInterface

const MOI = MathOptInterface

const CASE57_SOLUTION = [
    1.0260825400262428,
    0.0,
    0.6,
    0.0,
    8.603379123083476,
    0.0,
    1.3982297290334753,
    1.016218932360429,
    1.009467718490475,
    1.027857273911734,
    1.06,
    0.9962215167521776,
    0.9965415804936182
]

function moi_solve(
    optimizer::MOI.AbstractOptimizer,
    nlp::ExaPF.AbstractNLPEvaluator,
)
    block_data = MOI.NLPBlockData(nlp)
    u♭, u♯ = ExaPF.bounds(nlp, ExaPF.Variables())
    u0 = ExaPF.initial(nlp)
    n = ExaPF.n_variables(nlp)
    vars = MOI.add_variables(optimizer, n)
    # Set bounds and initial values
    for i in 1:n
        MOI.add_constraint(
            optimizer,
            MOI.SingleVariable(vars[i]),
            MOI.LessThan(u♯[i])
        )
        MOI.add_constraint(
            optimizer,
            MOI.SingleVariable(vars[i]),
            MOI.GreaterThan(u♭[i])
        )
        MOI.set(optimizer, MOI.VariablePrimalStart(), vars[i], u0[i])
    end
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer)
    x_opt = [MOI.get(optimizer, MOI.VariablePrimal(), v) for v in vars]
    solution = (
        minimum=MOI.get(optimizer, MOI.ObjectiveValue()),
        minimizer=x_opt
    )
    MOI.empty!(optimizer)
    return solution
end

@testset "MOI wrapper" begin
    datafile = joinpath(dirname(@__FILE__), "..", "data", "case57.m")
    nlp = ExaPF.ReducedSpaceEvaluator(datafile)
    optimizer = Ipopt.Optimizer(
        print_level=0,
        limited_memory_max_history=50,
        hessian_approximation="limited-memory",
        tol=1e-2,
    )
    solution = moi_solve(optimizer, nlp)
    @test solution.minimum ≈ 3.7589338e+04
    @test solution.minimizer == CASE57_SOLUTION
end
