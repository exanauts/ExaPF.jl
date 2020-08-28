using CUDA
using ExaPF
using KernelAbstractions
using Test
using TimerOutputs

import ExaPF: ParsePSSE, PowerSystem

@testset "Powerflow residuals and Jacobian" begin
    local_case = "case14.raw"
    # read data
    to = TimerOutputs.TimerOutput()
    datafile = joinpath(dirname(@__FILE__), "data", local_case)
    data_raw = ParsePSSE.parse_raw(datafile)
    data, bus_to_indexes = ParsePSSE.raw_to_exapf(data_raw)

    # Parsed data indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

    # retrive required data
    bus = data["bus"]
    gen = data["gen"]
    SBASE = data["baseMVA"][1]
    nbus = size(bus, 1)

    # obtain V0 from raw data
    V = Array{Complex{Float64}}(undef, nbus)
    T = Vector
    for i in 1:nbus
        V[i] = bus[i, VM]*exp(1im * pi/180 * bus[i, VA])
    end
    @test V ≈ Complex{Float64}[
        1.06 + 0.0im,
        1.0410510706561686 - 0.0907616013832108im,
        0.985192522040012 - 0.22247627854771523im,
        1.0012292929704543 - 0.18218707911892243im,
        1.0075796620614788 - 0.15551162239548505im,
        1.0372102511734809 - 0.2628590779498494im,
        1.0327942548732372 - 0.24527685887754397im,
        1.0605035588701377 - 0.2518575026156106im,
        1.0202428186152266 - 0.27219984563562466im,
        1.0147053262903118 - 0.27373721193522754im,
        1.0218895875940064 - 0.26981552747562876im,
        1.0188740342304141 - 0.27444787933420284im,
        1.0138437793219441 - 0.2746250817572887im,
        0.995247767507711 - 0.286014443990015im
    ]

    # form Y matrix
    Ybus = PowerSystem.makeYbus(data, bus_to_indexes);

    Vm = abs.(V)
    Va = angle.(V)
    bus = data["bus"]
    gen = data["gen"]
    nbus = size(bus, 1)
    ngen = size(gen, 1)

    ybus_re, ybus_im = ExaPF.Spmat{T}(Ybus)
    SBASE = data["baseMVA"][1]
    Sbus, Sload = PowerSystem.assembleSbus(gen, bus, SBASE, bus_to_indexes)
    pbus = real(Sbus)
    qbus = imag(Sbus)

    # Test that Sbus is correctly specified
    @test Sbus ≈ Complex{Float64}[
         2.32393 - 0.16549im,
         0.23189298795657234 + 0.3255213997211121im,
        -0.8066188947413118 + 0.028255632303189476im,
        -0.5123531308565288 + 0.035668879376258705im,
        -0.07125129097998142 - 0.02060016150325537im,
        -0.1098278333939612 + 0.07281082918372937im,
         0.0 + 0.0im,
         0.0 + 0.17623im,
        -0.24683805744186976 - 0.15824985834313557im,
        -0.08821656550979241 - 0.05553726513776928im,
        -0.038463023291667925 - 0.014532033764664084im,
        -0.05643415630748495 - 0.017413308476656675im,
        -0.11665673277410679 - 0.05281302692126483im,
        -0.13051302125309594 - 0.04527619677595794im
    ]

    ref, pv, pq = PowerSystem.bustypeindex(bus, gen, bus_to_indexes)
    npv = size(pv, 1)
    npq = size(pq, 1)

    @testset "Computing residuals" begin
        F = zeros(Float64, npv + 2*npq)
        # First compute a reference value for resisual computed at V
        F♯ = ExaPF.residualFunction(V, Ybus, Sbus, pv, pq)
        # residual_polar! uses only binary types as this function is meant
        # to be deported on the GPU
        ExaPF.residualFunction_polar!(F, Vm, Va,
            ybus_re, ybus_im,
            pbus, qbus, pv, pq, nbus)
        @test F ≈ F♯
    end
    @testset "Computing Jacobian of residuals" begin
        F = zeros(Float64, npv + 2*npq)
        # Compute Jacobian at point V manually and use it as reference
        J = ExaPF.residualJacobian(V, Ybus, pv, pq)
        J♯ = copy(J)

        # Then, create a JacobianAD object
        jacobianAD = ExaPF.AD.StateJacobianAD(F, Vm, Va,
                                              ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus)
        # and compute Jacobian with ForwardDiff
        ExaPF.AD.residualJacobianAD!(
            jacobianAD, ExaPF.residualFunction_polar!, Vm, Va,
            ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, to)
        @test jacobianAD.J ≈ J♯
    end
end
