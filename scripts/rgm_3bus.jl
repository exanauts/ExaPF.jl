using NLsolve
using ForwardDiff


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

function gfun(x, u, p, mode=0)

  if mode == 0
    F = zeros(3)
  elseif mode == 1
    F = similar(x)
  else
    F = similar(u)
  end

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

  cost = (w1*(5.0*VM1*VM3 + VM1*VM3*(-4*cos(VA13) + 5*sin(VA13))) +
          w2*P2)
  return cost
end


# OPF COMPUTATION

function solve_pf(x, u, p, verbose=true)
 
  println("Solving power flow")
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

# evaluate function
F = zeros(3)
gfun!(F, x, u, p)
println(F)

# solve power flow
xk = solve_pf(x, u, p)
println(xk)

# jacobian
gx_closure(x) = gfun(x, u, p, 1)
gx = x -> ForwardDiff.jacobian(gx_closure, x)

# gradient
cfun_closure(x) = cfun(x, u, p)
fx = x ->ForwardDiff.gradient(cfun_closure, x)
println(fx(xk))

# lamba calculation
lambda = -inv(gx(xk)')*fx(xk)
println(lambda)

# gradient of cost function
cfun_closure2(u) = cfun(x, u, p)
fu = u ->ForwardDiff.gradient(cfun_closure2, u)
gx_closure2(u) = gfun(x, u, p, 2)
gu = u -> ForwardDiff.jacobian(gx_closure2, u)

grad_c = fu(u) + gu(u)'*lambda

println(grad_c)

# step
println(u)
c_par = 0.1
unew = u - c_par*grad_c
println(unew)
