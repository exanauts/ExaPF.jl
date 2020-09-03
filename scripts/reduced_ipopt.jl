# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using Ipopt

import ExaPF: ParseMAT, PowerSystem, IndexSet

get_lines_limit(pf::PowerSystem.PowerNetwork) = (pf.data["branch"][:, 6] / pf.data["baseMVA"][1]).^2

function flow_limit(pf, x, u, p; T=Float64)
    nref = length(pf.ref)
    npv = length(pf.pv)
    npq = length(pf.pq)
    b2i = pf.bus_to_indexes
    branches = pf.data["branch"]
    nlines = size(branches, 1)
    cons_fr = zeros(T, nlines)
    cons_to = zeros(T, nlines)
    Vm, Va, pbus, qbus = ExaPF.PowerSystem.retrieve_physics(pf, x, u, p; V=T)

    for i in 1:nlines
        bus_fr = b2i[Int(branches[i, 1])]
        cons_fr[i] = pbus[bus_fr]^2 + qbus[bus_fr]^2
        bus_to = b2i[Int(branches[i, 2])]
        cons_to[i] = pbus[bus_to]^2 + qbus[bus_to]^2
    end
    return [cons_fr; cons_to]
end

# Build all the callbacks in a single closure.
function build_callback(form, x0, u, p, constraints)
    hash_u = hash(u)
    nlp = ExaPF.ReducedSpaceEvaluator(form, x0, u, p; constraints=constraints)
    function _update(u)
        # Update hash
        hash_u = hash(u)
        ExaPF.update!(nlp, u)
    end
    function eval_f(u)
        (hash_u != hash(u)) && _update(u)
        return ExaPF.objective(nlp, u)
    end
    function eval_grad_f(u, grad_f)
        (hash_u != hash(u)) && _update(u)
        ExaPF.gradient!(nlp, grad_f, u)
        return nothing
    end
    function eval_g(u, g)
        (hash_u != hash(u)) && _update(u)
        ExaPF.constraint!(nlp, g, u)
        return nothing
    end
    function eval_jac_g(u::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
        n = length(u)
        m = ExaPF.n_constraints(nlp)
        if mode == :Structure
            r, c = ExaPF.jacobian_structure!(nlp)
            rows .= r
            cols .= c
        else
            (hash_u != hash(u)) && _update(u)
            J = ExaPF.jacobian!(nlp, u)

            # @info("j", J)
            # Copy to Ipopt's Jacobian
            k = 1
            for i in 1:m
                for j in 1:n
                    values[k] = J[i, j]
                    k += 1
                end
            end
        end
        return nothing
    end
    function eval_h(u::Vector{Float64}, mode,
                    rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64,
                    lambda::Vector{Float64}, values::Vector{Float64})
        n = length(u)
        # if mode == :Structure
        #     index = 0
        #     for i in 1:n
        #         for j in i:n
        #             index += 1
        #             rows[index] = i
        #             cols[index] = j
        #         end
        #     end
        # else
        #     (hash_u != hash(u)) && _update(u)
        # end
        return nothing
    end
    return nlp, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h
end

function run_reduced_ipopt(datafile; hessian=false, cons=false)
    pf = PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, CPU())
    x0 = ExaPF.initial(polar, State())
    u0 = ExaPF.initial(polar, Control())
    p = ExaPF.initial(polar, Parameters())

    xk = copy(x0)
    uk = copy(u0)
    n = length(uk)

    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    # constraints = Function[]
    # Build callbacks in closure
    nlp, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h = build_callback(polar, xk, uk, p, constraints)
    m = ExaPF.n_constraints(nlp)

    # Set bounds on decision variable
    x_L = nlp.u_min
    x_U = nlp.u_max
    g_L = nlp.g_min
    g_U = nlp.g_max

    jnnz = n * m
    hnnz = 0

    # Number of nonzeros in upper triangular Hessian
    prob = Ipopt.createProblem(n, x_L, x_U, m, g_L, g_U, jnnz, hnnz,
                               eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

    prob.x = (nlp.u_min .+ nlp.u_max) ./ 2.0
    # prob.x = uk

    # This tests callbacks.
    function intermediate(alg_mod::Int, iter_count::Int,
                          obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
                          d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
                          ls_trials::Int)
        return iter_count < 500  # Interrupts after one iteration.
    end

    # I am too lazy to compute second order information
    if !hessian
        addOption(prob, "hessian_approximation", "limited-memory")
        addOption(prob, "limited_memory_initialization", "scalar1")
        addOption(prob, "limited_memory_max_history", 50)
        # addOption(prob, "limited_memory_update_type", "sr1")
    end
    # addOption(prob, "accept_after_max_steps", 10)
    addOption(prob, "tol", 1e-2)
    # addOption(prob, "derivative_test", "first-order")
    Ipopt.setIntermediateCallback(prob, intermediate)

    Ipopt.solveProblem(prob)
    u = prob.x
    # x, g, Jx, Ju, conv = ExaPF.solve(pf, xk, u, p, maxiter=50, tol=1e-14)
    ExaPF.PowerSystem.print_state(pf, nlp.x, u, p)
    # res = flow_limit(pf, x, u, p) .- [F_max; F_max]
    # println(res)
    # @info("l", maximum(res))
    # println(findall(x -> x > 0, res))
    return prob, pf
end

# datafile = "test/data/case9.m"
# datafile = "../pglib-opf/pglib_opf_case30_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case57_ieee.m"
datafile = "../pglib-opf/pglib_opf_case118_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case300_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case1354_pegase.m"
# datafile = "../pglib-opf/pglib_opf_case1888_rte.m"
# datafile = "../pglib-opf/pglib_opf_case200_activ.m"
# datafile = "../pglib-opf/pglib_opf_case500_goc.m"
prob, pf = run_reduced_ipopt(datafile; hessian=false, cons=false)
prob.status

