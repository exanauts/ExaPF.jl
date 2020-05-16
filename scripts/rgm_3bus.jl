using NLsolve
using ForwardDiff
using LinearAlgebra
using Printf
using LineSearches
using Plots

# ELEMENTAL FUNCTIONS

function gfun!(F, x, u, p)
  
  # retrieve variables
  VM3 = x[1]
  VA3 = x[2]
  VA2 = x[3]

  VM1 = u[1]
  P2 = u[2]
  VM2 = u[3]

  VA1 = p[1]
  P3 = p[2]
  Q3 = p[3]

  # intermediate quantities
  VA23 = VA2 - VA3
  VA31 = VA3 - VA1
  VA32 = VA3 - VA2

  F[1] = 4.0*VM2*VM2 + VM2*VM3*(-4*cos(VA23) + 10*sin(VA23)) - P2
  F[2] = (8.0*VM3*VM3 + VM3*VM1*(-4*cos(VA31) + 5*sin(VA31))
          + VM3*VM2*(-4*cos(VA32) + 10*sin(VA32)) + P3)
  F[3] = (15.0*VM3*VM3 + VM3*VM1*(-4*sin(VA31) - 5*cos(VA31))
          + VM3*VM2*(-4*sin(VA32) - 10*cos(VA32)) + Q3)
end

function gfun(x, u, p, T=typeof(x))
  # Get Float64 of Vectors{Float64}, that is the first parameter
  F = zeros(T.parameters[1], 3)

  # retrieve variables
  VM3 = x[1]
  VA3 = x[2]
  VA2 = x[3]

  VM1 = u[1]
  P2 = u[2]
  VM2 = u[3]

  VA1 = p[1]
  P3 = p[2]
  Q3 = p[3]

  # intermediate quantities
  VA23 = VA2 - VA3
  VA31 = VA3 - VA1
  VA32 = VA3 - VA2

  F[1] = 4.0*VM2*VM2 + VM2*VM3*(-4*cos(VA23) + 10*sin(VA23)) - P2
  F[2] = (8.0*VM3*VM3 + VM3*VM1*(-4*cos(VA31) + 5*sin(VA31))
          + VM3*VM2*(-4*cos(VA32) + 10*sin(VA32)) + P3)
  F[3] = (15.0*VM3*VM3 + VM3*VM1*(-4*sin(VA31) - 5*cos(VA31))
          + VM3*VM2*(-4*sin(VA32) - 10*cos(VA32)) + Q3)
  return F
end


function cfun(x, u, p)

  VM3 = x[1]
  VA3 = x[2]

  VM1 = u[1]
  P2 = u[2]

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

function pslack(x, u, p)

  VM3 = x[1]
  VA3 = x[2]

  VM1 = u[1]
  P2 = u[2]

  VA1 = p[1]

  VA13 = VA1 - VA3

  return (4.0*VM1*VM1 + VM1*VM3*(-4*cos(VA13) + 5*sin(VA13)))
end


# OPF COMPUTATION

function solve_pf(x, u, p, verbose=true)
 
  fun_pf!(F, x) = gfun!(F, x, u, p)
  x0 = copy(x)
  res = nlsolve(fun_pf!, x0)
  if verbose
    show(res)
    println("")
  end
  
  return res.zero

end

# initial parameters
x = zeros(3)
u = zeros(3)
p = zeros(3)

# this is an initial guess
x[1] = 1.0 #VM3
x[2] = 0.0 #VA3
x[3] = 0.0 #VA2

# this is given by the problem data, but might be "controlled" via OPF
u[1] = 1.0 #VM1
u[2] = 1.7 #P2
u[3] = 1.0 #VM2

# these parameters are fixed through the computation
p[1] = 0.0 #VA1, slack angle
p[2] = 2.0 #P3
p[3] = 1.0 #Q3



# print initial guesses
println(x)
println(u)

# copy to iteration vectors
xk = copy(x)
uk = copy(u)
s = similar(uk)

iterations = 0

ls = BackTracking()


for i = 1:100
  global xk
  global uk
  println("Iteration ", i)

  # solve power flow
  println("Solving power flow")
  xk = solve_pf(xk, uk, p, false)

  # jacobian
  gx_x(x) = gfun(x, uk, p, typeof(x))
  gx = x -> ForwardDiff.jacobian(gx_x, x)

  # gradient
  cfun_x(x) = cfun(x, uk, p)
  fx = x ->ForwardDiff.gradient(cfun_x, x)

  # lamba calculation
  println("Computing Lagrange multipliers")
  lambda = -inv(gx(xk)')*fx(xk)

  # compute f_u, g_u
  cfun_u(u) = cfun(xk, u, p)
  fu = u ->ForwardDiff.gradient(cfun_u, u)
  gx_u(u) = gfun(xk, u, p, typeof(u))
  gu = u -> ForwardDiff.jacobian(gx_u, u)

  # compute gradient of cost function
  Lu = u -> cfun_u(u) + gx_u(u)'*lambda
  grad_Lu = u -> (fu(u) + gu(u)'*lambda)
  grad_L = grad_Lu(uk)
  println("fu: ", norm(fu(uk)))
  println("gu(uk)'*lambda: ", norm(gu(uk)'*lambda))
  println("lambda: ", lambda)
  println("Norm of gradient ", norm(grad_L))

  # Search/step direction
  global s .= -grad_L
  Lalpha(alpha) = Lu(uk .+ alpha.*s)
  function grad_Lalpha(alpha)
    return dot(grad_Lu(uk .+ alpha .* s), s)
  end
  function Lgrad_Lalpha(alpha)
    gvec = grad_Lu(uk .+ alpha .* s)
    phi = Lu(uk .+ alpha .*s)
    dphi = dot(gvec, s)
    return (phi, dphi)
  end
  dL_0 = dot(s, grad_L)
  # step
  println("Computing new control vector")
  alpha = 0.1
  obj = Lu(uk)
  # alpha, obj = ls(Lalpha, grad_Lalpha, Lgrad_Lalpha, 1.0, obj, dL_0)
  uk = uk - alpha*grad_L
  
  println("Cost: ", cfun(xk, uk, p))
  #@printf("VM3 %3.2f. VA3 %2.2f. VA2 %2.2f.\n", xk[1], xk[2], xk[3])
  #@printf("VM1 %3.2f. P2 %2.2f. VM2 %2.2f.\n", uk[1], uk[2], uk[3])

end
