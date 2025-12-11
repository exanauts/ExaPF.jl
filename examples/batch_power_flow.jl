using ExaPF

using LinearAlgebra
const PS = ExaPF.PowerSystem

# Example of creating loads for multiple scenarios
# ploads, and qloads are matrices of size (nbus, nscen)
# This creates a equal loads across all scenarios for simplicity
function create_loads(form, nscen::Int)
    nbus = ExaPF.PowerSystem.get(form, ExaPF.PowerSystem.NumberOfBuses())
    stack = ExaPF.NetworkStack(form)
    pload = stack.params[1:nbus]
    qload = stack.params[nbus+1:2*nbus]
    ploads = zeros(Float64, nbus, nscen)
    qloads = zeros(Float64, nbus, nscen)
    for i in 1:nscen
        ploads[:,i] .= pload
    end
    for i in 1:nscen
        qloads[:,i] .= qload
    end
    return ploads, qloads
end

# 10 scenarios
nscen = 10
backend = CPU()  # Replace with CUDABackend() to run on NVIDIA GPU
case = "case9.m"
# Load case to use load data from the file
polar = ExaPF.load_polar(case)
ploads, qloads = create_loads(polar, nscen)
println("Solve $nscen scenarios for case $case with on $backend)")
res = run_pf(case, backend, :block_polar, nscen, ploads, qloads; verbose=2)
sol_cpu = get_solution(res)
