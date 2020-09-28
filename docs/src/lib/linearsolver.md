```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    const Precondition = ExaPF.Precondition
    const Iterative = ExaPF.Iterative
end
DocTestFilters = [r"ExaPF"]
```

## Description
```@docs
Precondition.AbstractPreconditioner
```

## API Reference
```@docs
Precondition.Preconditioner
Precondition.update
Precondition.build_adjmatrix
Precondition.fillblock_gpu!
Precondition.fillP_gpu!
```
