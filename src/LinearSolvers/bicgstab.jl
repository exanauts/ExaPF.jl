"""
    bicgstab(A, b, P, xi;
             tol=1e-8,
             maxiter=size(A, 1),
             verbose=false,
             maxtol=1e20)

BiCGSTAB implementation according to

> Van der Vorst, Henk A.
> "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems."
> SIAM Journal on scientific and Statistical Computing 13, no. 2 (1992): 631-644.

"""
function bicgstab(A, b, P, xi;
                  tol=1e-8, maxiter=size(A, 1), verbose=false, maxtol=1e20)
    # parameters
    n    = size(b, 1)
    mul!(xi, P, b)
    ri   = b - A * xi
    br0  = copy(ri)
    rho0 = 1.0
    alpha = 1.0
    omega0 = 1.0
    vi = similar(xi)
    pi = similar(xi)
    fill!(vi, 0)
    fill!(pi, 0)

    rhoi   = copy(rho0)
    omegai = copy(omega0)
    residual = copy(b)

    y = similar(pi)
    s = similar(pi)
    z = similar(pi)
    t1 = similar(pi)
    t2 = similar(pi)
    pi1 = similar(pi)
    vi1 = similar(pi)
    t = similar(pi)

    go = true
    status = Unsolved
    iter = 1
    restarts = 0
    while go
        rhoi1 = dot(br0, ri) ;
        if abs(rhoi1) < 1e-20
            restarts += 1
            ri .= b
            mul!(ri, A, xi, -1.0, 1.0)
            br0 .= ri
            residual .= b
            rho0 = 1.0
            rhoi = rho0
            rhoi1 = dot(br0,ri)
            alpha = 1.0
            omega0 = 1.0
            omegai = 1.0
            fill!(vi, 0.0)
            fill!(pi, 0.0)
        end
        beta = (rhoi1/rhoi) * (alpha / omegai)
        pi1 .= ri .+ beta .* (pi .- omegai .* vi)
        mul!(y, P, pi1)
        mul!(vi1, A, y)
        alpha = rhoi1 / dot(br0, vi1)
        s .= ri .- (alpha .* vi1)

        mul!(z, P, s)
        mul!(t, A, z)
        mul!(t1, P, t)
        mul!(t2, P, s)
        omegai1 = dot(t1, t2) / dot(t1, t1)
        xi .= xi .+ alpha .* y .+ omegai1 .* z

        mul!(residual, A, xi, 1.0, -1.0)
        anorm = xnorm(residual)

        if verbose
            @printf("%4d %10.4e\n", iter, anorm)
        end

        if isnan(anorm)
            go = false
            status = NotANumber
        end
        if anorm < tol
            go = false
            status = Converged
            verbose && println("Tolerance reached at iteration $iter")
        elseif anorm > maxtol
            go = false
            status = Diverged
        elseif maxiter == iter
            go = false
            status = MaxIterations
            verbose && println("Not converged")
        end

        ri     .= s .- omegai1 .* t
        rhoi   = rhoi1
        pi     .= pi1
        vi     .= vi1
        omegai = omegai1
        iter   += 1
    end

    return xi, iter, status
end
