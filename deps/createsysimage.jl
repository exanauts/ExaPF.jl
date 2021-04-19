using PackageCompiler
@warn "Manually uncomment MOI test in runtests.jl. This test breaks PackageCompiler."
create_sysimage([:CUDA, :ExaPF]; sysimage_path="exapf.so", precompile_execution_file="test/runtests.jl")
