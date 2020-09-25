# ExaPF

[ExaPF.jl](https://github.com/exanauts/ExaPF.jl) is a
package to solve powerflow problem on exascale architecture. `ExaPF.jl` aims to
implement a reduced method for solving the optimal power flow problem (OPF)
fully on GPUs. Reduced methods enforce the constraints, represented here by
the power flow's (PF) system of nonlinear equations, separately at each
iteration of the optimization in the reduced space. This paper describes the
API of `ExaPF.jl` for solving the power flow's nonlinear equations (NLE) entirely on the GPU.
This includes the computation of the derivatives using automatic
differentiation, an iterative linear solver with a preconditioner, and a
Newton-Raphson implementation. All of these steps allow us to run the main
computational loop entirely on the GPU with no transfer from host to device.

This implementation will serve as the basis for the future optimal power flow (OPF) implementation as a nonlinear programming problem (NLP)
in the reduced space.

To make our implementation portable to CPU and GPU architectures we leverage
two abstractions: arrays and kernels. Both of these abstractions are
supported through the packages `CUDA.jl` [@besard2017juliagpu; @besard2019prototyping] and `KernelAbstractions.jl`
Please take a look at the [autodiff](autodiff.md) and [linear solver](linearsolver.md) 
implementations to get a design overview of `ExaPF.jl` targeted for GPUs. 

The user API is separated into three layers:

* First layer or physical layer: Power network topology in [powersystem](powersystem.md) 
* Second layer: Interface between power network and NLE or NLP in [formulations](formulations.md) 
* Third layer: Evaluators for NLE or NLP 

The third layer is for users working in optimization whereas the first layer is for electrical engineers. They meet in the second layer.

## Table of contents

```@contents
Pages = ["autodiff.md", "linearsolver.md", "powersystem.md", "formulations.md", "evaluators.md"]
Depth = 3
```