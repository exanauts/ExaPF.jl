module ExaBenchmark

using Printf
# GPU
using KernelAbstractions

# Algorithms
using Krylov
using ExaPF

const LS = ExaPF.LinearSolvers
const PS = ExaPF.PowerSystem

const CONFIG_FILE = joinpath(dirname(@__FILE__), "config.json")
const OUTPUT_DIR = joinpath(dirname(@__FILE__), "results")

DEFAULT_CONFIG = Dict{Symbol, Any}(
    :ntrials_callbacks=>100,
    :ntrials_powerflow=>2,
    :ntrials_iterative=>2,
    :overlaps=>[0],
    :size_blocks=>[32,64,128,256,512],
    :npartitions=>64,
    :noverlaps=>0,
)

function write_csv(output, results)
    io = open(output, "w")
    names = results[:id]
    times = results[:time]
    iters = results[:iters]
    @printf(io, "id,iters,time\n")
    for i in eachindex(names)
        @printf(
            io,
            "%s,%d,%.5f\n",
            names[i],
            iters[i],
            times[i],
        )
    end
    close(io)
end

function benchmark_expressions(polar, config)
    ntrials = config[:ntrials_callbacks]::Int
    # Init
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    # Build expressions tree
    expr = ExaPF.MultiExpressions([
        ExaPF.PowerFlowBalance(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]) ∘ basis
    n = length(expr)
    output = similar(stack.input, n)
    # Precompilation
    expr(output, stack)
    # Timings
    total_time = 0.0
    for i in 1:ntrials
        total_time += @elapsed begin
            expr(output, stack)
        end
    end
    total_time /= ntrials
    return (name="callbacks_expr", time=total_time, it=0)
end

function benchmark_adjoint(polar, config)
    ntrials = config[:ntrials_callbacks]::Int
    # Init
    stack = ExaPF.NetworkStack(polar)
    ∂stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    # Build expressions tree
    expr = ExaPF.MultiExpressions([
        ExaPF.PowerFlowBalance(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]) ∘ basis
    n = length(expr)
    output = similar(stack.input, n)
    v = similar(stack.input, n) ; fill!(v, 1)
    # Warm-up
    ExaPF.adjoint!(expr, ∂stack, stack, v)
    # Timings
    total_time = 0.0
    for i in 1:ntrials
        total_time += @elapsed begin
            ExaPF.adjoint!(expr, ∂stack, stack, v)
        end
    end
    total_time /= ntrials
    return (name="callbacks_adjoint", time=total_time, it=0)
end

function benchmark_jacobian_powerflow(polar, config)
    ntrials = config[:ntrials_callbacks]::Int
    # Init
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    pf = ExaPF.PowerFlowBalance(polar) ∘ basis
    jac = ExaPF.Jacobian(polar, pf, State())
    # Warm-up
    ExaPF.jacobian!(jac, stack)
    # Timings
    total_time = 0.0
    for i in 1:ntrials
        total_time += @elapsed begin
            ExaPF.jacobian!(jac, stack)
        end
    end
    total_time /= ntrials
    return (name="callbacks_jacobian_powerflow", time=total_time, it=0)
end

function benchmark_jacobian_all(polar, config)
    ntrials = config[:ntrials_callbacks]::Int
    # Init
    stack = ExaPF.NetworkStack(polar)
    ∂stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    # Build expressions tree
    expr = ExaPF.MultiExpressions([
        ExaPF.PowerFlowBalance(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]) ∘ basis
    maps = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]
    jac = ExaPF.Jacobian(polar, expr, maps)
    # Warm-up
    ExaPF.jacobian!(jac, stack)
    # Timings
    total_time = 0.0
    for i in 1:ntrials
        total_time += @elapsed begin
            ExaPF.jacobian!(jac, stack)
        end
    end
    total_time /= ntrials
    return (name="callbacks_jacobian_all", time=total_time, it=0)
end

function benchmark_hessian_lagrangian(polar, config)
    ntrials = config[:ntrials_callbacks]::Int
    # Init
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    # Build expressions tree
    expr = ExaPF.MultiExpressions([
        ExaPF.CostFunction(polar),
        ExaPF.PowerFlowBalance(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]) ∘ basis
    n = length(expr)
    maps = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]
    hess = ExaPF.FullHessian(polar, expr, maps)
    y = similar(stack.input, n) ; fill!(y, 1)
    # Warm-up
    ExaPF.hessian!(hess, stack, y)
    # Timings
    total_time = 0.0
    for i in 1:ntrials
        total_time += @elapsed begin
            ExaPF.hessian!(hess, stack, y)
        end
    end
    total_time /= ntrials
    return (name="callbacks_hessian_lagrangian", time=total_time, it=0)
end

function benchmark_powerflow(polar, config, linear_algo)
    ntrials = config[:ntrials_powerflow]::Int
    noverlaps = config[:noverlaps]::Int
    npartitions = config[:npartitions]::Int
    # Init
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    pflow = ExaPF.PowerFlowBalance(polar) ∘ basis
    jx = ExaPF.Jacobian(polar, pflow ∘ basis, State())
    # Build preconditioner
    J = jx.J
    n = size(J, 1)
    precond = LS.BlockJacobiPreconditioner(J, npartitions, polar.device, noverlaps)

    algo = linear_algo(J; P=precond)
    powerflow_solver = NewtonRaphson(tol=1e-8)
    VT = typeof(stack.input)
    pf_buffer = ExaPF.NLBuffer{VT}(n)

    # Warm-up
    conv = ExaPF.nlsolve!(
        powerflow_solver, jx, stack; linear_solver=algo, nl_buffer=pf_buffer,
    )
    niters = conv.n_iterations

    total_time = 0.0
    for i in 1:ntrials
        ExaPF.init!(polar, stack)
        total_time += @elapsed begin
            res = @timed ExaPF.nlsolve!(powerflow_solver, jx, stack; linear_solver=algo, nl_buffer=pf_buffer,
    )
        end
        conv = res.value
        @assert conv.n_iterations == niters
    end
    total_time /= ntrials
    return (name="powerflow_iterative", time=total_time, it=niters)
end

function benchmark_bicgstab(polar, config, noverlaps, nblocks)
    ntrials = config[:ntrials_iterative]::Int
    # Init
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    pflow = ExaPF.PowerFlowBalance(polar) ∘ basis
    jx = ExaPF.Jacobian(polar, pflow ∘ basis, State())
    # Evaluate Jacobian
    ExaPF.jacobian!(jx, stack)
    # Build preconditioner
    J = jx.J
    n = size(J, 1)
    npartitions = max(ceil(Int, n / nblocks), 2)
    precond = LS.BlockJacobiPreconditioner(J, npartitions, polar.device, noverlaps)
    algo = LS.Bicgstab(J; P=precond)
    # Update preconditioner
    LS.update!(algo, J)
    # RHS
    b = pflow(stack)
    x = copy(b)
    # Warm-up
    res = Krylov.bicgstab!(
        algo.inner, J, x;
        N=algo.precond,
        atol=algo.atol,
        rtol=algo.rtol,
        verbose=algo.verbose,
        history=true,
    )
    niters = length(res.stats.residuals)

    total_time = 0.0
    for i in 1:ntrials
        copyto!(x, b)
        total_time += @elapsed begin
            res = Krylov.bicgstab!(
                algo.inner, J, x;
                N=algo.precond,
                atol=algo.atol,
                rtol=algo.rtol,
                verbose=algo.verbose,
                history=true,
            )
            @assert niters == length(res.stats.residuals)
        end
    end
    return (name="bicgstab_$(nblocks)blk_$(noverlaps)overlap", time=total_time, it=niters)
end

function run_benchmarks_callbacks(polar, config=DEFAULT_CONFIG)
    names = String[]
    timings = Float64[]
    iters = Int[]

    @info("Benchmark callbacks")
    for bench in (
        benchmark_expressions,
        benchmark_adjoint,
        benchmark_jacobian_powerflow,
        benchmark_jacobian_all,
        benchmark_hessian_lagrangian,
    )
        res = bench(polar, DEFAULT_CONFIG)
        push!(names, res.name)
        push!(timings, res.time)
        push!(iters, res.it)
    end
    # Run powerflow with direct solver
    res = benchmark_powerflow(polar, config, LS.DirectSolver)
    push!(names, "powerflow_direct")
    push!(timings, res.time)
    push!(iters, res.it)

    return Dict(:id=>names, :time=>timings, :iters=>iters,)
end

function run_benchmarks_bicgstab(polar, config=DEFAULT_CONFIG)
    nx = ExaPF.number(polar, State())
    names = String[]
    timings = Float64[]
    iters = Int[]
    ref_overlap = Int[]
    ref_nblocks = Int[]
    @info("Benchmark BICGSTAB")
    overlaps = config[:overlaps]
    size_blocks = config[:size_blocks]

    for noverlap in overlaps, nblocks in size_blocks
        res = benchmark_bicgstab(polar, config, noverlap, nblocks)
        push!(names, res.name)
        push!(timings, res.time)
        push!(iters, res.it)
        push!(ref_overlap, noverlap)
        push!(ref_nblocks, nblocks)
    end
    # Run powerflow with iterative solver
    _, best_time = findmin(timings)

    config_pf = copy(config)
    olevel = ref_overlap[best_time]
    nblocks = ref_nblocks[best_time]
    config_pf[:noverlaps] = olevel
    config_pf[:npartitions] = ceil(Int, nx / nblocks)
    res = benchmark_powerflow(polar, config, LS.Bicgstab)
    push!(names, "powerflow_bicgstab_$(nblocks)blk_$(olevel)overlap")
    push!(timings, res.time)
    push!(iters, res.it)

    return Dict(:id=>names, :time=>timings, :iters=>iters,)
end


# Code taken from MathOptInterface.jl
#
# https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/Test.jl
#
# The MathOptInterface.jl package is licensed under the MIT "Expat" License:
#
#     Copyright (c) 2017: Miles Lubin and contributors Copyright (c) 2017: Google Inc.
#
# Complete license available here: https://github.com/jump-dev/MathOptInterface.jl/blob/master/LICENSE.md
function benchmark(
    casename, device;
    exclude=[], config=DEFAULT_CONFIG, outputdir=OUTPUT_DIR,
)
    if !isdir(outputdir)
        mkdir(outputdir)
    end
    println("BENCHMARK $casename on $(device)")
    polar = PolarForm(PS.load_case(casename), device)
    if isa(device, CPU)
        ext = ".cpu.csv"
    elseif isa(device, CUDABackend)
        ext = ".cuda.csv"
    end

    for name_sym in names(@__MODULE__; all = true)
        name = string(name_sym)
        if !startswith(name, "run_benchmarks_")
            continue
        elseif !isempty(exclude) && any(s -> occursin(s, name), exclude)
            continue
        end

        bench_function = getfield(@__MODULE__, name_sym)

        res = bench_function(polar, config)

        bench_name = split(name, '_')[end]
        instance_name = split(casename, '.')[1]
        outputfile = joinpath(outputdir, "$(bench_name)_$(instance_name)$(ext)")
        write_csv(outputfile, res)
    end
end

function default_benchmark(config::Dict)
    iterator = Any[("cpu", CPU())]
    if CUDA.has_cuda_gpu()
        push!(iterator, ("cuda", CUDABackend()))
    end

    for (kdev, device) in iterator
        for case in keys(config[kdev])
            conf = copy(DEFAULT_CONFIG)
            conf[:overlaps] = config[kdev][case]["overlaps"]
            exclude = config[kdev][case]["exclude"]
            benchmark(case, device; config=conf, exclude=exclude)
        end
    end
end

end
