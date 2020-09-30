```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    const AD = ExaPF.AD
end
DocTestFilters = [r"ExaPF"]
```

## Description
```@docs
AD.AbstractADFramework
```

## API Reference
```@docs
AD.StateJacobianAD
AD.DesignJacobianAD
AD.myseed_kernel_cpu
AD.myseed_kernel_gpu
AD.seeding
AD.getpartials_cpu
AD.getpartials_gpu
AD.getpartials
AD._uncompress
AD.uncompress!
AD.residualJacobianAD!
```
