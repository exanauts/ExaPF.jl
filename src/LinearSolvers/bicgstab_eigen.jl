# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
    bicgstab_eigen 

BiCGSTAB implementation exactly corresponding to the implementation in Eigen
"""
function bicgstab_eigen(A, b, P, x;
                  tol=1e-8, maxiter=size(A, 1), verbose=false)

    mul!(x, P, b)
    r  = b .- A * x
    r0 = similar(r)
    r0 .= r

    r0_sqnorm = norm(r0)^2
    rhs_sqnorm = norm(b)^2
    if rhs_sqnorm == 0
        x .= 0.0
        return x, 0, Converged
    end
    rho    = 1.0
    alpha  = 1.0
    w      = 1.0

    v = similar(x); p = similar(x)
    fill!(v, 0.0); fill!(p, 0.0)

    y = similar(x); z = similar(x)
    s = similar(x); t = similar(x)

    tol2 = tol*tol*rhs_sqnorm
    eps2 = eps(Float64)^2
    i = 0
    restarts = 0
    status = Unsolved

    while norm(r)^2 > tol2 && i < maxiter
        rho_old = rho

        rho = dot(r0, r)

        if abs(rho) < eps2*r0_sqnorm
            mul!(r, A, x)
            r .= b .- r
            r0 .= r
            v .= 0.0
            p .= 0.0
            rho = norm(r)^2
            r0_sqnorm = norm(r)^2
            if restarts == 0
                i = 0
            end
            restarts += 1
        end

        beta = (rho/rho_old) * (alpha / w)
        p .= r .+ (beta * (p .- w .* v))

        mul!(y, P, p)
        mul!(v, A, y)

        alpha = rho / dot(r0, v)
        s .= r .- alpha .* v

        mul!(z, P, s)
        mul!(t, A, z)

        tmp = norm(t)^2
        if tmp > 0.0
            w = dot(t,s) / tmp
        else
            w = 0.0
        end
        x .= x .+ alpha * y .+ w * z;
        r .= s .- w * t;
        i += 1
    end
    if maxiter == i
        go = false
        status = MaxIterations
        verbose && println("Restarts: $restarts")
        verbose && println("Not converged")
    end
    if norm(r)^2 <= tol2
        go = false
        status = Converged
        verbose && println("Restarts: $restarts")
        verbose && println("Tolerance reached at iteration $i")
    end
    return x, i, status
end
