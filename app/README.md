# Build ExaPF Binary using JuliaC

- Install `juliac` with

```julia
] app add JuliaC
```

- Instantiate the `app/ExaPFApp` environment

- From the `app` folder, run

```bash
juliac ExaPFApp --output-exe ExaPFBin --bundle exapfdir --experimental
```
