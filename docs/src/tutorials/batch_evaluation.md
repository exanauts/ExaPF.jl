```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    const LS = ExaPF.LinearSolvers
    const AD = ExaPF.AD
end
DocTestFilters = [r"ExaPF"]
```

```@setup batch_pf
using ExaPF
using KLU
using LinearAlgebra
const LS = ExaPF.LinearSolvers
const PS = ExaPF.PowerSystem

polar = ExaPF.load_polar("case9.m")
```


# Batch power flow

ExaPF provides a way to evaluate the expressions by blocks,
opening the way to introduce more parallelism in the code.

## BlockNetworkStack

We recall that a given [`NetworkStack`](@ref) `stack` stores the different
variables and parameters (power generations, voltages, loads) required to
evaluate the power flow model.
```@example batch_pf
stack = ExaPF.NetworkStack(polar);

```
The variables are stored in the field `stack.input`, the parameters
in the field `stack.params`. The parameters encode the active `pd` and
reactive loads `qd` at all buses in the network, such that
```@example batch_pf
nbus = ExaPF.get(polar, PS.NumberOfBuses());
pd = stack.params[1:nbus]
qd = stack.params[nbus+1:2*nbus]

```
In short, a [`NetworkStack`](@ref) stores one set of loads $$p_0$$.
On its side, the [`BlockNetworkStack`](@ref) extends the `NetworkStack` structure
to store $$N$$ different set of parameters (=scenarios) $$p_1, \cdots, p_N$$.
As an illustration, the syntax to instantiate a [`BlockNetworkStack`](@ref)
with $$N=10$$ different scenarios is:
```@example batch_pf
nscen = 10;
ploads = rand(nbus, nscen);
qloads = rand(nbus, nscen);
blk_stack = ExaPF.BlockNetworkStack(polar, ploads, qloads)

```
For each scenario $$i=1, \cdots, N$$, a `BlockNetworkStack` structure
stores the associated set of variables. As a consequence,
we get $$N$$ different realizations for the variables stored in the field `input`
(`vmag`, `vang` and `pgen`).
By default, the initial values are set according to the values
specified in `polar` (usually defined when importing the data from the instance file):
```@example batch_pf
reshape(blk_stack.vmag, nbus, nscen)
```
Only the parameters are varying according to the scenarios we passed as input
in the constructor:
```@example batch_pf
reshape(blk_stack.pload, nbus, nscen)
```


## Evaluate expressions in block

A [`BlockNetworkStack`](@ref) has the same behavior as a
[`NetworkStack`](@ref) structure, and any
[`AutoDiff.AbstractExpression`](@ref) can take a [`BlockNetworkStack`](@ref)
as input. ExaPF can takes advantage of the block structure
when using a [`BlockNetworkStack`](@ref).

As an example, suppose we want to evaluate the power flow
balances in block with a [`PowerFlowBalance`](@ref) expression:
```@example batch_pf
powerflow = ExaPF.PowerFlowBalance(polar) ∘ ExaPF.PolarBasis(polar);

```
A single evaluation takes as input the [`NetworkStack`](@ref) `stack` structure:
```@example batch_pf
output = powerflow(stack)

```
A block evaluation takes as input the [`BlockNetworkStack`](@ref) `blk_stack` structure:
```@example batch_pf
m = length(powerflow);
blk_output = zeros(m * nscen);
powerflow(blk_output, blk_stack); # inplace evaluation
reshape(blk_output, m, nscen)

```
We get $$N$$ different results for the power flow balance equations,
depending on which scenario we are on.


## Solve power flow in block on the CPU
Once the different structures used for block evaluation instantiated,
one is able to solve the power flow in block on the CPU using
the same function [`nlsolve!`](@ref). The block Jacobian is evaluated
with automatic differentiation using a `ArrowheadJacobian` structure:
```@example batch_pf
blk_jx = ExaPF.ArrowheadJacobian(polar, powerflow, State(), nscen);
blk_jx.J
```
We notice that the `ArrowheadJacobian` computes the resulting Jacobian
as a block diagonal matrix. The `ArrowheadJacobian` has a slightly
different behavior than its classical counterpart `AutoDiff.Jacobian`,
in the sense that one has to pass the parameters manually to initiate internally the
dual numbers:
```@example batch_pf
ExaPF.set_params!(blk_jx, blk_stack);
ExaPF.jacobian!(blk_jx, blk_stack);

```
As soon as the `blk_jx` initialized, we can solve the power flow
equations in block as
```@example batch_pf
conv = ExaPF.nlsolve!(
    NewtonRaphson(verbose=2),
    blk_jx,
    blk_stack;
)
```
At the solution, we get different values for the voltage magnitudes
at the PQ nodes:
```@example batch_pf
reshape(blk_stack.vmag, nbus, nscen)
```

## Solve power flow in batch on the GPU

When the [`BlockNetworkStack`](@ref) is instantiated on the GPU,
the expressions are evaluated in batch.
The syntax to solve the power flow equations is exactly the same as on the
CPU, using `cusolverRF` to solve the different linear systems:
```@example batch_pf
using CUSOLVERRF, CUDAKernels
polar_gpu = PolarForm(polar, CUDADevice()); # load model on GPU
blk_stack_gpu = ExaPF.BlockNetworkStack(polar_gpu, ploads, qloads);
powerflow_gpu = ExaPF.PowerFlowBalance(polar_gpu) ∘ ExaPF.PolarBasis(polar_gpu);
blk_jx_gpu = ExaPF.ArrowheadJacobian(polar_gpu, powerflow_gpu, State(), nscen);
ExaPF.set_params!(blk_jx_gpu, blk_stack_gpu);
ExaPF.jacobian!(blk_jx_gpu, blk_stack_gpu);
rf_fac = CUSOLVERRF.RFLU(blk_jx_gpu.J)
rf_solver = LS.DirectSolver(rf_fac)
conv = ExaPF.nlsolve!(
    NewtonRaphson(verbose=2),
    blk_jx_gpu,
    blk_stack_gpu;
    linear_solver=rf_solver,
)

```
