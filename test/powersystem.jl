using CUDA
using ExaPF
using KernelAbstractions
using Test
using TimerOutputs

import ExaPF: ParsePSSE, PowerSystem, IndexSet

const PS = PowerSystem

@testset "Powerflow residuals and Jacobian" begin
    local_case = "case14.raw"
    # read data
    to = TimerOutputs.TimerOutput()
    datafile = joinpath(dirname(@__FILE__), "..", "data", local_case)
    data_raw = ParsePSSE.parse_raw(datafile)
    data = ParsePSSE.raw_to_exapf(data_raw)

    # Parsed data indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()

    # retrive required data
    bus = data["bus"]
    gen = data["gen"]
    SBASE = data["baseMVA"][1]

    bus_to_indexes = PS.get_bus_id_to_indexes(bus)
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
    Ybus = PS.makeYbus(data, bus_to_indexes).ybus;

    Vm = abs.(V)
    Va = angle.(V)
    bus = data["bus"]
    gen = data["gen"]
    nbus = size(bus, 1)
    ngen = size(gen, 1)

    ybus_re, ybus_im = ExaPF.Spmat{T{Int}, T{Float64}}(Ybus)
    SBASE = data["baseMVA"][1]
    Sbus, Sload = PS.assembleSbus(gen, bus, SBASE, bus_to_indexes)
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

    ref, pv, pq = PS.bustypeindex(bus, gen, bus_to_indexes)
    npv = size(pv, 1)
    npq = size(pq, 1)

    @testset "Computing residuals" begin
        F = zeros(Float64, npv + 2*npq)
        # First compute a reference value for resisual computed at V
        F♯ = ExaPF.power_balance(V, Ybus, Sbus, pv, pq)
        # residual_polar! uses only binary types as this function is meant
        # to be deported on the GPU
        ExaPF.residual_polar!(F, Vm, Va,
            ybus_re, ybus_im,
            pbus, qbus, pv, pq, nbus)
        @test F ≈ F♯
    end
    @testset "Computing Jacobian of residuals" begin
        F = zeros(Float64, npv + 2*npq)
        # Compute Jacobian at point V manually and use it as reference
        J = ExaPF.residual_jacobian(State(), V, Ybus, pv, pq, ref)
        J♯ = copy(J)
        # Build the Jacobian structure
        mappv = [i + nbus for i in pv]
        mappq = [i + nbus for i in pq]
        # Ordering for x is (θ_pv, θ_pq, v_pq)
        statemap = vcat(mappv, mappq, pq)
        jacobian_structure = ExaPF.StateJacobianStructure{Vector{Float64}}(ExaPF.residual_jacobian, statemap)

        # Then, create a Jacobian object
        jacobianAD = ExaPF.AutoDiff.Jacobian(jacobian_structure, F, Vm, Va,
                                              ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, ExaPF.AutoDiff.StateJacobian())
        # and compute Jacobian with ForwardDiff
        ExaPF.AutoDiff.residual_jacobian!(
            jacobianAD, ExaPF.residual_polar!, Vm, Va,
            ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, ExaPF.AutoDiff.StateJacobian())
        @test jacobianAD.J ≈ J♯
    end
end

@testset "PowerNetwork object" begin
    psse_datafile = "case14.raw"
    matpower_datafile = "case9.m"

    # Test constructor
    @testset "Parsers $name" for name in [psse_datafile, matpower_datafile]
        datafile = joinpath(dirname(@__FILE__), "..", "data", name)
        pf = PS.PowerNetwork(datafile)
        @test isa(pf, PS.PowerNetwork)
    end

    # From now on, test with "case9.m"
    datafile = joinpath(dirname(@__FILE__), "..", "data", matpower_datafile)
    data = PS.import_dataset(datafile)
    pf = PS.PowerNetwork(data)

    @testset "Computing cost coefficients" begin
        coefs = PS.get_costs_coefficients(pf)
        @test size(coefs) == (3, 4)
        @test isequal(coefs[:, 1], [3.0, 2.0, 2.0])
    end

    @testset "Getters" for Attr in [
        PS.NumberOfBuses,
        PS.NumberOfPVBuses,
        PS.NumberOfPQBuses,
        PS.NumberOfSlackBuses,
        PS.NumberOfLines,
        PS.NumberOfGenerators,
    ]
        res = PS.get(pf, Attr())
        @test isa(res, Int)
    end

    @testset "Indexing" begin
        idx = PS.get(pf, PS.GeneratorIndexes())
        @test isequal(idx, [1, 2, 3])
    end

    @testset "Bounds" begin
        n_bus = PS.get(pf, PS.NumberOfBuses())
        v_min, v_max = PS.bounds(pf, PS.Buses(), PS.VoltageMagnitude())
        @test length(v_min) == n_bus
        @test length(v_max) == n_bus

        n_gen = PS.get(pf, PS.NumberOfGenerators())
        p_min, p_max = PS.bounds(pf, PS.Generator(), PS.ActivePower())
        @test length(p_min) == n_gen
        @test length(p_max) == n_gen
        q_min, q_max = PS.bounds(pf, PS.Generator(), PS.ReactivePower())
        @test length(q_min) == n_gen
        @test length(q_max) == n_gen

        n_lines = PS.get(pf, PS.NumberOfLines())
        f_min, f_max = PS.bounds(pf, PS.Lines(), PS.ActivePower())
        @test length(f_min) == n_lines
        @test length(f_max) == n_lines
    end

    @testset "Load from data" begin
        pf_original = PS.PowerNetwork(data)
        @test isa(pf_original, PS.PowerNetwork)
        n_lines = PS.get(pf_original, PS.NumberOfLines())

        pf_removed = PS.PowerNetwork(data; remove_lines=Int[1])
        @test isa(pf_removed, PS.PowerNetwork)
        n_after_removal = PS.get(pf_removed, PS.NumberOfLines())

        @test n_lines - 1 == n_after_removal
        @test pf_original.Ybus != pf_removed.Ybus
    end
end
