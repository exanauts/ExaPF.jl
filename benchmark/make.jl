
using Pkg
using JSON

const CONFIG_FILE = joinpath(dirname(@__FILE__), "config.json")

benchmark_dir = dirname(@__FILE__)
# Use latest version of ExaPF
Pkg.develop(PackageSpec(path=joinpath(benchmark_dir, "..")))
Pkg.instantiate()

# Config
println("Config file: ", CONFIG_FILE)
config = JSON.parsefile(CONFIG_FILE)

# Benchmark
include(joinpath(benchmark_dir, "benchmarks.jl"))
ExaBenchmark.default_benchmark(config)
