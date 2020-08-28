using CUDA
using CUDA.CUSPARSE
using ExaPF
using KernelAbstractions
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using TimerOutputs
import ExaPF: PowerSystem, IndexSet, AD

# read data
function run_level0(datafile; device=CPU())
    to = TimerOutputs.TimerOutput()
    pf = PowerSystem.PowerNetwork(datafile, 1)

    # retrieve initial state of network
    pbus = real.(pf.sbus)
    qbus = imag.(pf.sbus)
    Vm = abs.(pf.vbus)
    Va = angle.(pf.vbus)
    nbus = length(pbus)

    ref = pf.ref
    pv = pf.pv
    pq = pf.pq
    npv = size(pv, 1);
    npq = size(pq, 1);
    ybus_re, ybus_im = ExaPF.Spmat{Array}(pf.Ybus)

    F = zeros(Float64, npv + 2*npq)
    @info("Benchmark function residualFunction_polar!")
    @btime begin
        ExaPF.residualFunction_polar!($F, $Vm, $Va,
            $ybus_re, $ybus_im,
            $pbus, $qbus, $pv, $pq, $nbus)
    end

    jacobianAD = AD.StateJacobianAD(F, Vm, Va,
                                    ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus)
    @info("Benchmark residualJacobianAD!")
    @btime begin
        AD.residualJacobianAD!($jacobianAD, ExaPF.residualFunction_polar!,
                               $Vm, $Va,
                               $ybus_re, $ybus_im, $pbus, $qbus, $pv, $pq, $ref, $nbus, $to)
    end
    return
end

function run_level1(datafile; device=CPU())
    pf = PowerSystem.PowerNetwork(datafile, 1)

    # retrieve initial state of network
    pbus = real.(pf.sbus)
    qbus = imag.(pf.sbus)
    vmag = abs.(pf.vbus)
    vang = angle.(pf.vbus)

    x = ExaPF.PowerSystem.get_x(pf, vmag, vang, pbus, qbus)
    u = ExaPF.PowerSystem.get_u(pf, vmag, vang, pbus, qbus)
    p = ExaPF.PowerSystem.get_p(pf, vmag, vang, pbus, qbus)

    n = length(u)
    @btime begin
        ExaPF.solve($pf, $x, $u, $p, tol=1e-14, verbose_level=0)
    end
    ExaPF.solve(pf, x, u, p, tol=1e-14, verbose_level=2)
    return
end

datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
run_level0(datafile)
