using NLsolve

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
          + VM3*VM2*(4*cos(VA32) + 10*sin(VA32)) + P3)
  F[3] = (15.0*VM3*VM3 + VM3*VM1*(-4*sin(VA31) - 5*cos(VA31))
          + VM3*VM2*(-4*sin(VA32) - 10*cos(VA32)) + Q3)
end

function solve_pf(x, u, p)
  
  fun_pf!(F, x) = gfun!(F, x, u, p)
  x0 = copy(x)
  res = nlsolve(fun_pf!, x0)

  show(res)
  println("")
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
solve_pf(x, u, p)
