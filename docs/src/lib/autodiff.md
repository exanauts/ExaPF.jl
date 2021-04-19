```@meta
CurrentModule = ExaPF.AutoDiff
```
# AutoDiff

## Adjoint

```@docs
TapeMemory
```

## Jacobian

```@docs
AbstractJacobian
ConstantJacobian
Jacobian
jacobian!
```

### API reference for the Jacobian
```@docs
seed!
getpartials_kernel!
uncompress_kernel!
```

## Hessian
```@docs
AbstractHessian
Hessian
adj_hessian_prod!
```
