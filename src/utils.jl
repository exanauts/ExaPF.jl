struct ConvergenceStatus
    has_converged::Bool
    n_iterations::Int
    norm_residuals::Float64
    n_linear_solves::Int
end

mutable struct Spmat{VTI<:AbstractVector, VTF<:AbstractVector}
    colptr::VTI
    rowval::VTI
    nzval::VTF

    # create 2 Spmats from complex matrix
    function Spmat{VTI, VTF}(mat::SparseMatrixCSC{Complex{Float64}, Int}) where {VTI, VTF}
        matreal = new(VTI(mat.colptr), VTI(mat.rowval), VTF(real.(mat.nzval)))
        matimag = new(VTI(mat.colptr), VTI(mat.rowval), VTF(imag.(mat.nzval)))
        return matreal, matimag
    end
    # copy constructor
    function Spmat{VTI, VTF}(mat) where {VTI, VTF}
        return new(VTI(mat.colptr), VTI(mat.rowval), VTF(mat.nzval))
    end
end

# small utils function
function polar!(Vm, Va, V, ::CPU)
    Vm .= abs.(V)
    Va .= angle.(V)
end
function polar!(Vm, Va, V, ::CUDADevice)
    Vm .= CUDA.abs.(V)
    Va .= CUDA.angle.(V)
end

# norm
norm2(x::AbstractVector) = norm(x, 2)
norm2(x::CuVector) = CUBLAS.nrm2(x)

function project_constraints!(u::AbstractArray, grad::AbstractArray, u_min::AbstractArray,
                              u_max::AbstractArray)
    dim = length(u)
    for i in 1:dim
        if u[i] > u_max[i]
            u[i] = u_max[i]
            grad[i] = 0.0
        elseif u[i] < u_min[i]
            u[i] = u_min[i]
            grad[i] = 0.0
        end
    end
end
