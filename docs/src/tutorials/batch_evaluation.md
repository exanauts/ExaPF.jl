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

## BlockPolarForm

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
By default, a [`NetworkStack`](@ref) stores one set of loads $$p_0$$.

Suppose now we want to evaluate the model associated with the polar
formulation for $$N$$ different set of parameters (=scenarios) $$p_1, \cdots, p_N$$.
ExaPF allows to streamline the polar formulation with a [`BlockPolarForm`](@ref)
structure:
```@example batch_pf
nscen = 10;
blk_polar = ExaPF.BlockPolarForm(polar, nscen)

```
Then, ExaPF can also instantiate a [`NetworkStack`](@ref)
object, with the memory required to store the variables of
the different scenarios:
```@example batch_pf
blk_stack = ExaPF.NetworkStack(blk_polar)

```
We can pass the scenarios manually using the function
`set_params!`:
```@example batch_pf

ploads = rand(nbus, nscen);
qloads = rand(nbus, nscen);
ExaPF.set_params!(blk_stack, ploads, qloads)

```
The structure `blk_stack` stores
$$N$$ different realizations for the variables stored in the field `input`
(`vmag`, `vang` and `pgen`).
By default, the initial values are set according to the values
specified in `blk_polar` (usually defined when importing the data from the instance file):
```@example batch_pf
reshape(blk_stack.vmag, nbus, nscen)
```
Only the parameters are varying according to the scenarios we passed as input
in the constructor:
```@example batch_pf
reshape(blk_stack.pload, nbus, nscen)
```


## Evaluate expressions in block

ExaPF takes advantage of the block structure when using a [`BlockPolarForm`](@ref).

As an example, suppose we want to evaluate the power flow
balances in block form with a [`PowerFlowBalance`](@ref) expression:
```@example batch_pf
powerflow = ExaPF.PowerFlowBalance(blk_polar) ∘ ExaPF.PolarBasis(blk_polar);

```
A block evaluation takes as input the [`NetworkStack`](@ref) `blk_stack` structure:
```@example batch_pf
m = div(length(powerflow), nscen);
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
with automatic differentiation using a `BatchJacobian` structure:
```@example batch_pf
blk_jx = ExaPF.BatchJacobian(blk_polar, powerflow, State());
blk_jx.J
```
We notice that the `BatchJacobian` computes the resulting Jacobian
as a block diagonal matrix. The `BatchJacobian` has a slightly
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

When the [`BlockPolarForm`](@ref) model is instantiated on the GPU,
the expressions are evaluated in batch.
The syntax to solve the power flow equations is exactly the same as on the
CPU, using `cuDSS` to solve the different linear systems:
```@example batch_pf
using CUDA
using CUDSS
polar_gpu = ExaPF.load_polar("case9.m", CUDABackend());
blk_polar_gpu = ExaPF.BlockPolarForm(polar_gpu, nscen); # load model on GPU
blk_stack_gpu = ExaPF.NetworkStack(blk_polar_gpu);
ExaPF.set_params!(blk_stack_gpu, ploads, qloads);
powerflow_gpu = ExaPF.PowerFlowBalance(blk_polar_gpu) ∘ ExaPF.PolarBasis(blk_polar_gpu);
blk_jx_gpu = ExaPF.BatchJacobian(blk_polar_gpu, powerflow_gpu, State());
ExaPF.set_params!(blk_jx_gpu, blk_stack_gpu);
ExaPF.jacobian!(blk_jx_gpu, blk_stack_gpu);
rf_fac = CUDSS.lu(blk_jx_gpu.J)
rf_solver = LS.DirectSolver(rf_fac)
conv = ExaPF.nlsolve!(
    NewtonRaphson(verbose=2),
    blk_jx_gpu,
    blk_stack_gpu;
    linear_solver=rf_solver,
)

```
