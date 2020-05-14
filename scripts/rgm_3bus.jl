using NLsolve
using ForwardDiff
using LinearAlgebra
using Printf

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
global xk = copy(x)
global uk = copy(u)

iterations = 0


for i = 1:100
  global xk
  global uk
  println("Iteration ", i)

  # solve power flow
  println("Solving power flow")
  xk = solve_pf(xk, uk, p, false)

  # jacobian
  gx_closure(x) = gfun(x, u, p, typeof(x))
  gx = x -> ForwardDiff.jacobian(gx_closure, x)

  # gradient
  cfun_closure(x) = cfun(x, u, p)
  fx = x ->ForwardDiff.gradient(cfun_closure, x)

  # lamba calculation
  println("Computing lagrange multipliers")
  lambda = -inv(gx(xk)')*fx(xk)

  # compute g_u, g_u
  cfun_closure2(u) = cfun(x, u, p)
  fu = u ->ForwardDiff.gradient(cfun_closure2, u)
  gx_closure2(u) = gfun(x, u, p, typeof(u))
  gu = u -> ForwardDiff.jacobian(gx_closure2, u)

  # compute gradient of cost function
  grad_c = fu(uk) + gu(uk)'*lambda
  println("Norm of gradient ", norm(grad_c))

  # step
  println("Computing new control vector")
  c_par = 0.1
  uk = uk - c_par*grad_c
  
  println("Cost: ", cfun(xk, uk, p))
  #@printf("VM3 %3.2f. VA3 %2.2f. VA2 %2.2f.\n", xk[1], xk[2], xk[3])
  #@printf("VM1 %3.2f. P2 %2.2f. VM2 %2.2f.\n", uk[1], uk[2], uk[3])

end
