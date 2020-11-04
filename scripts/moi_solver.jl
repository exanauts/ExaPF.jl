
using Ipopt
using MathOptInterface

const MOI = MathOptInterface

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
end

datafile = joinpath(dirname(@__FILE__), "..", "test", "data", "case57.m")

nlp = ExaPF.ReducedSpaceEvaluator(datafile)
aug = ExaPF.AugLagEvaluator(nlp, ExaPF.initial(nlp); c₀=0.1, scale=true)
# optimizer = KNITRO.Optimizer(hessopt=3)
optimizer = Ipopt.Optimizer(limited_memory_max_history=50, tol=1e-2)

moi_solve(optimizer, aug)

