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
    ploads = repeat(pload, 1, nscen)
    qloads = repeat(qload, 1, nscen)
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
res = run_pf(case, backend, :block_polar, nscen, ploads, qloads)
sol = get_solution(res)

# Compare all scenarios solution norm against single scenario solution norm
res_single = run_pf(case, backend, :polar)
sol_single = get_solution(res_single)
nstates = length(sol_single)  # Number of state variables per scenario

println("Size of single scenario solution: ", size(sol_single))
println("Size of batched scenarios solution: ", size(sol))

# Check if all scenarios match the single scenario solution norm
match = all(i -> isapprox(norm(sol[(i-1)*nstates + 1 : i*nstates]), norm(sol_single)), 1:nscen)

if match
    println("Batched power flow solutions match single scenario solution norm")
else
    error("Batched power flow solutions do not match single scenario solution norm")
end
