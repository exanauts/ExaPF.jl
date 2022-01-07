
struct NLBuffer{VT}
    x::VT
    y::VT
end
NLBuffer{VT}(n::Int) where VT = NLBuffer(VT(undef, n), VT(undef, n))

function extract_values!(dest, src)
    @assert length(dest) == length(src)
    for i in eachindex(dest)
        dest[i] = src[i].value
    end
end

function nlsolve!(
    algo::NewtonRaphson,
    jac::MyJacobian,
    state::NetworkStack{VT,Buf};
    linear_solver=DirectSolver(),
    nl_buffer=NLBuffer{VT}(size(jac, 2)),
) where {VT, Buf}
    iter = 0
    converged = false
    normF = Inf
    linsol_iters = Int[]

    map = jac.map
    x = view(state.input, map)

    residual = nl_buffer.y
    Δx = nl_buffer.x

    for i in 1:algo.maxiter
        J = jacobian!(jac, state)
        extract_values!(residual, jac.t1sF)

        normF = xnorm(residual)
        if xnorm(residual) < algo.tol
            converged = true
            break
        end

        # Update
        n_iters = LS.ldiv!(linear_solver, Δx, J, residual)
        x .= x .- Δx

        push!(linsol_iters, n_iters)

        iter += 1
    end
    return ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
end
