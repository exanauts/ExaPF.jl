
"""
residualFunction

Assembly residual function for N-R power flow
"""
function residualFunction(V, Ybus, Sbus, pv, pq)
    # form mismatch vector
    mis = V .* conj(Ybus * V) - Sbus
    # form residual vector
    F = [real(mis[pv])
         real(mis[pq])
         imag(mis[pq]) ]
    return F
end

function residualJacobian(V, Ybus, pv, pq)
    n = size(V, 1)
    Ibus = Ybus*V
    diagV       = sparse(1:n, 1:n, V, n, n)
    diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
    diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

    dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)

    j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
    j12 = real(dSbus_dVm[[pv; pq], pq])
    j21 = imag(dSbus_dVa[pq, [pv; pq]])
    j22 = imag(dSbus_dVm[pq, pq])

    J = [j11 j12; j21 j22]
end

@kernel function residual_kernel!(F, @Const(v_m), @Const(v_a),
                                  @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval),
                                  @Const(ybus_im_nzval), @Const(ybus_im_colptr), @Const(ybus_im_rowval),
                                  @Const(pinj), @Const(qinj), @Const(pv), @Const(pq), nbus)

    npv = size(pv, 1)
    npq = size(pq, 1)

    i = @index(Global, Linear)
    # REAL PV: 1:npv
    # REAL PQ: (npv+1:npv+npq)
    # IMAG PQ: (npv+npq+1:npv+2npq)
    fr = (i <= npv) ? pv[i] : pq[i - npv]
    F[i] -= pinj[fr]
    if i > npv
        F[i + npq] -= qinj[fr]
    end
    @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
        to = ybus_re_rowval[c]
        aij = v_a[fr] - v_a[to]
        # f_re = a * cos + b * sin
        # f_im = a * sin - b * cos
        coef_cos = v_m[fr]*v_m[to]*ybus_re_nzval[c]
        coef_sin = v_m[fr]*v_m[to]*ybus_im_nzval[c]
        cos_val = cos(aij)
        sin_val = sin(aij)
        F[i] += coef_cos * cos_val + coef_sin * sin_val
        if i > npv
            F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
        end
    end
end

function residualFunction_polar!(F, v_m, v_a,
                                 ybus_re, ybus_im,
                                 pinj, qinj, pv, pq, nbus)
    npv = length(pv)
    npq = length(pq)
    if isa(F, Array)
        kernel! = residual_kernel!(CPU(), 4)
    else
        kernel! = residual_kernel!(CUDADevice(), 256)
    end
    ev = kernel!(F, v_m, v_a,
                 ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
                 ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
                 pinj, qinj, pv, pq, nbus,
                 ndrange=npv+npq)
    wait(ev)
end

# TODO: kernalize
function transfer!(polar::PolarForm, cache::NetworkState, x, u, p)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq

    i_pq = 1
    for bus in pq
        cache.vang[bus] = x[npv+i_pq]
        cache.vmag[bus] = x[npv+npq+i_pq]
        cache.pinj[bus] = p[nref + i_pq]
        cache.qinj[bus] = p[nref + npq + i_pq]
        i_pq += 1
    end

    i_ref = 1
    for bus in ref
        cache.pinj[bus] = PS.get_power_injection(bus, cache.vmag, cache.vang, polar.ybus_re, polar.ybus_im)
        cache.qinj[bus] = PS.get_react_injection(bus, cache.vmag, cache.vang, polar.ybus_re, polar.ybus_im)
        cache.vmag[bus] = u[i_ref]
        cache.vang[bus] = p[i_ref]
        i_ref += 1
    end

    i_pv = 1
    for bus in pv
        cache.qinj[bus] = PS.get_react_injection(bus, cache.vmag, cache.vang, polar.ybus_re, polar.ybus_im)
        cache.vang[bus] = x[i_pv]
        cache.pinj[bus] = u[nref + i_pv] - polar.active_load[bus]
        cache.vmag[bus] = u[nref + npv + i_pv]
        i_pv += 1
    end
end

# TODO: kernalize
function refresh!(polar::PolarForm, ::PS.Generator, ::PS.ActivePower, cache::NetworkState)
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    index_gen = polar.indexing.index_generators
    index_ref = polar.indexing.index_ref
    index_pv = polar.indexing.index_pv
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen

    # TODO: check the complexity of this for loop
    for i in 1:npv
        bus = index_pv[i]
        i_gen = pv_to_gen[i]
        cache.pg[i_gen] = cache.pinj[bus] + polar.active_load[bus]
    end
    for i in 1:nref
        bus = index_ref[i]
        i_gen = ref_to_gen[i]
        inj = PS.get_power_injection(bus, cache.vmag, cache.vang, polar.ybus_re, polar.ybus_im)
        cache.pg[i_gen] = inj + polar.active_load[bus]
    end
end