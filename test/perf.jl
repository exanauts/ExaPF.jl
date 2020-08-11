target = "cpu"
using CUDA
using CUDA.CUSPARSE
using ExaPF
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using TimerOutputs
import ExaPF: ParsePSSE, PowerSystem, IdxSet

# read data
function run_benchmark(datafile)
    to = TimerOutputs.TimerOutput()
    data_raw = ParsePSSE.parse_raw(datafile)
    data = ParsePSSE.raw_to_exapf(data_raw)
    
    # Parsed data indexes
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, BUS_VM, BUS_VA, BASE_KV, ZONE, VMAX, VMIN,
    LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IdxSet.idx_bus()

    # retrive required data
    bus = data["bus"]
    gen = data["gen"]
    SBASE = data["baseMVA"][1]
    nbus = size(bus, 1)

    # obtain V0 from raw data
    V = Array{Complex{Float64}}(undef, nbus)
    T = Vector
    for i in 1:nbus
        V[i] = bus[i, BUS_VM]*exp(1im * pi/180 * bus[i, BUS_VA])
    end

    # form Y matrix
    Ybus = PowerSystem.makeYbus(data);

    Vm = abs.(V)
    Va = angle.(V)
    bus = data["bus"]
    gen = data["gen"]
    SBASE = data["baseMVA"][1]
    nbus = size(bus, 1)

    ybus_re, ybus_im = ExaPF.Spmat{T}(Ybus)
    SBASE = data["baseMVA"][1]
    Sbus = PowerSystem.assembleSbus(gen, bus, SBASE)
    pbus = real(Sbus)
    qbus = imag(Sbus)

    ref, pv, pq = PowerSystem.bustypeindex(bus, gen)
    npv = size(pv, 1);
    npq = size(pq, 1);

    F = zeros(Float64, npv + 2*npq)
    println("[CPU] Benchmark function residualFunction_polar!")
    @btime begin
        ExaPF.residualFunction_polar!($F, $Vm, $Va,
            $ybus_re.nzval, $ybus_re.colptr, $ybus_re.rowval,
            $ybus_im.nzval, $ybus_im.colptr, $ybus_im.rowval,
            $pbus, $qbus, $pv, $pq, $nbus)
    end

    J = ExaPF.residualJacobian(V, Ybus, pv, pq)
    F = zeros(Float64, npv + 2*npq)

    # Then, create a JacobianAD object
    coloring = ExaPF.matrix_colors(J)
    jacobianAD = ExaPF.AD.JacobianAD(J, coloring, F, Vm, Va, pv, pq)
    # and compute Jacobian with ForwardDiff
    println("[CPU] Benchmark residualJacobianAD!")
    @btime begin
        ExaPF.AD.residualJacobianAD!(
            $jacobianAD, ExaPF.residualFunction_polar!, $Vm, $Va,
            $ybus_re, $ybus_im, $pbus, $qbus, $pv, $pq, $nbus, $to)
    end

    # Passing to GPU
    target = "cuda"
    gF = F |> CuArray
    gVm = Vm |> CuArray
    gVa = Va |> CuArray
    gpbus = pbus |> CuArray
    gqbus = qbus |> CuArray
    gpv = pv |> CuArray
    gpq = pq |> CuArray
    gybus_re, gybus_im = ExaPF.Spmat{CuVector}(Ybus)
    fill!(gF, 0.0)

    println("[GPU] Benchmark residualFunction_polar!")
    @btime begin
        $gF .= .0
        ExaPF.residualFunction_polar!($gF, $gVm, $gVa,
                $gybus_re.nzval, $gybus_re.colptr, $gybus_re.rowval,
                $gybus_im.nzval, $gybus_im.colptr, $gybus_im.rowval,
                $gpbus, $gqbus, $gpv, $gpq, $nbus)
    end
    gJ = CuSparseMatrixCSR(J)

    # Then, create a JacobianAD object
    ccolors = ExaPF.matrix_colors(J) |> CuArray

    gjacobianAD = ExaPF.AD.JacobianAD(gJ, ccolors, gF, gVm, gVa, gpv, gpq)
    println("[GPU] Benchmark residualJacobianAD!")
    @btime begin
        ExaPF.AD.residualJacobianAD!(
            $gjacobianAD, ExaPF.residualFunction_polar!, $gVm, $gVa,
            $gybus_re, $gybus_im, $gpbus, $gqbus, $gpv, $gpq, $nbus, $to)
    end
end

datafile = joinpath(dirname(@__FILE__), "case14.raw")
run_benchmark(datafile)
