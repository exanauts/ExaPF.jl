  
using Ipopt
using ForwardDiff

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)

function eval_f(x)
  VM3 = x[1]
  VA3 = x[2]

  VM1 = x[4]
  P2 = x[5]

  VA1 = p[1]

  VA13 = VA1 - VA3

  # we fix generation weights inside the
  # function to simplify the script and
  # follow the paper closely.
  w1 = 1.0
  w2 = 1.0

  cost = (w1*(4.0*VM1*VM1 + VM1*VM3*(-4*cos(VA13) + 5*sin(VA13))) +
          w2*P2)
  return cost
end

function eval_g(x, g)
  # Bad: g    = zeros(2)  # Allocates new array
  # OK:  g[:] = zeros(2)  # Modifies 'in place'

  # Get Float64 of Vectors{Float64}, that is the first parameter

  # retrieve variables
  VM3 = x[1]
  VA3 = x[2]
  VA2 = x[3]

  VM1 = x[4]
  P2 = x[5]
  VM2 = x[6]

  VA1 = p[1]
  P3 = p[2]
  Q3 = p[3]

  # intermediate quantities
  VA23 = VA2 - VA3
  VA31 = VA3 - VA1
  VA32 = VA3 - VA2

  g[1] = 4.0*VM2*VM2 + VM2*VM3*(-4*cos(VA23) + 10*sin(VA23)) - P2
  g[2] = (8.0*VM3*VM3 + VM3*VM1*(-4*cos(VA31) + 5*sin(VA31))
          + VM3*VM2*(-4*cos(VA32) + 10*sin(VA32)) + P3)
  g[3] = (15.0*VM3*VM3 + VM3*VM1*(-4*sin(VA31) - 5*cos(VA31))
          + VM3*VM2*(-4*sin(VA32) - 10*cos(VA32)) + Q3)
end

function eval_g(x)
  T = typeof(x)
  g = zeros(T.parameters[1], 3)
  eval_g(x,g)
  return g
end

function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
  # Bad: grad_f    = zeros(4)  # Allocates new array
  # OK:  grad_f[:] = zeros(4)  # Modifies 'in place'
  ForwardDiff.gradient!(grad_f, eval_f, x)
end

function eval_jac_g(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
  if mode == :Structure
    idx = 1
    for c in 1:m #number of constraints
        for i in 1:n # number of variables
            rows[idx] = c ; cols[idx] = i
            idx += 1 
        end
    end
  else
    jac = ForwardDiff.jacobian(eval_g, x)
    k = 1
    for i in 1:m
        for j in 1:n
            values[k] = jac[i,j]
            k += 1
        end
    end
  end
end

function eval_h(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Vector{Float64})
  if mode == :Structure
    # Symmetric matrix, fill the lower left triangle only
    idx = 1
    for row = 1:6
      for col = 1:row
        rows[idx] = row
        cols[idx] = col
        idx += 1
      end
    end
  else
    hf = ForwardDiff.hessian(eval_f, x)
    k = 1
    for i in 1:n
        for j in 1:i
            values[k] = obj_factor * hf[i,j] 
            k += 1
        end
    end
    # hg = ForwardDiff.hessian(eval_g, x)
    h = x -> ForwardDiff.jacobian(x -> ForwardDiff.jacobian(eval_g, x), x)
    hg = h(x)
    hg = reshape(hg, m, n, n)
    for l in 1:m
        k = 1
        for i in 1:n
            for j in 1:i
            values[k] += lambda[l] * hg[l,i,j]
            k += 1
            end
        end
    end
  end
end

n = 6
x_L = [-1e18,-1e18,-1e18,-1e18,-1e18,-1e18]
x_U = [1e18,1e18,1e18,1e18,1e18,1e18]

m = 3
g_L = [0.0, 0.0, 0.0]
g_U = [0.0, 0.0, 0.0]

p = zeros(3)
p[1] = 0.0 #VA1, slack angle
p[2] = 2.0 #P3
p[3] = 1.0 #Q3

# Number of nonzeros in upper triangular Hessian
hnnz = Int(n*(n+1)/2)
prob = createProblem(n, x_L, x_U, m, g_L, g_U, 18, hnnz,
                     eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

prob.x = [1.0, 0.0, 0.0, 1.0, 1.7, 1.0]

# This tests callbacks.
function intermediate(alg_mod::Int, iter_count::Int,
  obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
  ls_trials::Int)
  return iter_count < 100  # Interrupts after one iteration.
end

setIntermediateCallback(prob, intermediate)

solvestat = solveProblem(prob)
