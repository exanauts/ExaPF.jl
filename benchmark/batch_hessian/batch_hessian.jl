using BlockPowerFlow.CUSOLVERRF

struct BatchHessianStorage{VT, MT, Hess, Fac1, Fac2}
    nbatch::Int
    state::Hess
    # Batch factorization
    ∇g::Fac1
    ∇gᵀ::Fac2
    # RHS
    ∂f::VT
    ∂²f::VT
    # Adjoints
    z::MT
    ψ::MT
    tmp_tgt::MT
    tmp_hv::MT
end

function BatchHessianStorage(polar::PolarForm{T, VI, VT, MT}, J, nbatch::Int) where {T, VI, VT, MT}
    nx = get(polar, NumberOfState())
    nu = get(polar, NumberOfControl())
    ngen = get(polar, PS.NumberOfGenerators())
    Hstate = ExaPF.batch_hessian(polar, ExaPF.power_balance, nbatch)
    ∂f = VT(undef, ngen)
    ∂²f = VT(undef, ngen)

    z = MT(undef, nx, nbatch)
    ψ = MT(undef, nx, nbatch)
    tgt = MT(undef, nx+nu, nbatch)
    hv =  MT(undef, nx+nu, nbatch)

    if isa(polar.device, CPU)
        ∇g = lu(J)
        ∇gᵀ = ∇g'
    else
        ∇g = CUSOLVERRF.CusolverRfLUBatch(J, nbatch; fast_mode=true)
        Jᵀ = CUSPARSE.CuSparseMatrixCSC(J)
        ∇gᵀ = CUSOLVERRF.CusolverRfLUBatch(Jᵀ, nbatch; fast_mode=true)
    end

    return BatchHessianStorage(nbatch, Hstate, ∇g, ∇gᵀ, ∂f, ∂²f, z, ψ, tgt, hv)
end

function batch_hessprod!(nlp::ExaPF.ReducedSpaceEvaluator, batch_ad, hessmat, u, w)
    @assert nlp.hessians != nothing

    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    ∇gᵤ = nlp.state_jacobian.u.J

    nbatch = batch_ad.nbatch
    # Second order adjoints
    ψ = batch_ad.ψ
    z = batch_ad.z
    # Two vector products
    tgt = batch_ad.tmp_tgt
    hv = batch_ad.tmp_hv

    mul!(z, ∇gᵤ, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(batch_ad.∇g, z)

    # Init tangent
    for i in 1:nbatch
        mt = 1 + (i-1)*(nx+nu)
        mz = 1 + (i-1)*nx
        copyto!(tgt, mt, z, mz, nx)
        mw = 1 + (i-1)*nu
        copyto!(tgt, mt+nx, w, mw, nu)
    end

    ## OBJECTIVE HESSIAN
    # TODO: not implemented yet
    ∂fₓ = hv[1:nx, :]
    ∂fᵤ = hv[nx+1:nx+nu, :]

    ExaPF.batch_adj_hessian_prod!(nlp.model, batch_ad.state, hv, nlp.buffer, nlp.λ, tgt)
    ∂fₓ .= @view hv[1:nx, :]
    ∂fᵤ .= @view hv[nx+1:nx+nu, :]

    # Second order adjoint
    LinearAlgebra.ldiv!(ψ, batch_ad.∇gᵀ, ∂fₓ)

    hessmat .= ∂fᵤ
    mul!(hessmat, transpose(∇gᵤ), ψ, -1.0, 1.0)

    return
end

function fast_hessprod!(nlp::ExaPF.ReducedSpaceEvaluator, ∇g, hessvec, u::Array, w::Array)
    @assert nlp.hessians != nothing

    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    ∇gᵤ = nlp.state_jacobian.u.J

    H = nlp.hessians
    # Second order adjoints
    ψ = H.ψ
    z = H.z
    # Two vector products
    tgt = H.tmp_tgt
    hv = H.tmp_hv

    mul!(z, ∇gᵤ, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(∇g, z)

    # Init tangent
    copyto!(tgt, 1, z, 1, nx)
    copyto!(tgt, 1, w, 1, nu)

    ## OBJECTIVE HESSIAN
    # TODO: not implemented yet
    ∂fₓ = hv[1:nx]
    ∂fᵤ = hv[nx+1:nx+nu]

    AutoDiff.adj_hessian_prod!(nlp.model, H.state, hv, nlp.buffer, nlp.λ, tgt)
    ∂fₓ .= @view hv[1:nx]
    ∂fᵤ .= @view hv[nx+1:nx+nu]

    # Second order adjoint
    LinearAlgebra.ldiv!(ψ, ∇g', ∂fₓ)

    hessvec .= ∂fᵤ
    mul!(hessvec, transpose(∇gᵤ), ψ, -1.0, 1.0)

    return
end

function cpu_hessian!(nlp::ExaPF.AbstractNLPEvaluator, hess, x)
    n = ExaPF.n_variables(nlp)
    J = nlp.state_jacobian.x.J
    ∇g = lu(J)
    v = similar(x)
    tic = time()
    @inbounds for i in 1:n
        hv = @view hess[:, i]
        fill!(v, 0)
        v[i] = 1.0
        fast_hessprod!(nlp, ∇g, hv, x, v)
    end
    println("Elapsed: ", time() - tic)
end

