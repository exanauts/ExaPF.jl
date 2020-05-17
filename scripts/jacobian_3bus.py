from sympy import *

VM3      = symbols('VM3')
VA3      = symbols('VA3')
VA2      = symbols('VA2')

VM1      = symbols('VM1')
P2      = symbols('P2')
VM2      = symbols('VM2')

VA1      = symbols('VA1')
P3      = symbols('P3')
Q3      = symbols('Q3')

w1      = symbols('w1')
w2      = symbols('w2')

# intermediate quantities
VA23 = VA2 - VA3
VA31 = VA3 - VA1
VA32 = VA3 - VA2
VA13 = VA1 - VA3

# EQUATIONS PFLOW

F1 = 4.0*VM2*VM2 + VM2*VM3*(-4*cos(VA23) + 10*sin(VA23)) - P2
F2 = (8.0*VM3*VM3 + VM3*VM1*(-4*cos(VA31) + 5*sin(VA31))
      + VM3*VM2*(-4*cos(VA32) + 10*sin(VA32)) + P3)
F3 = (15.0*VM3*VM3 + VM3*VM1*(-4*sin(VA31) - 5*cos(VA31))
      + VM3*VM2*(-4*sin(VA32) - 10*cos(VA32)) + Q3)

# EQUATION COST FUNCTION

cost = (w1*(4.0*VM1*VM1 + VM1*VM3*(-4*cos(VA13) + 5*sin(VA13))) +
          w2*P2)

print("POWER FLOW JACOBIAN w.r.t. X")
print("#F1")
print("J[0, 0] = ", diff(F1, VM3))
print("J[0, 1] = ", diff(F1, VA3))
print("J[0, 2] = ", diff(F1, VA2))
print("#F2")
print("J[1, 0] = ", diff(F2, VM3))
print("J[1, 1] = ", diff(F2, VA3))
print("J[1, 2] = ", diff(F2, VA2))
print("#F3")
print("J[2, 0] = ", diff(F3, VM3))
print("J[2, 1] = ", diff(F3, VA3))
print("J[2, 2] = ", diff(F3, VA2))
print("")

print("POWER FLOW JACOBIAN w.r.t. U")
print("#F1")
print("J[0, 0] = ", diff(F1, VM1))
print("J[0, 1] = ", diff(F1, P2))
print("J[0, 2] = ", diff(F1, VM2))
print("#F2")
print("J[1, 0] = ", diff(F2, VM1))
print("J[1, 1] = ", diff(F2, P2))
print("J[1, 2] = ", diff(F2, VM2))
print("#F3")
print("J[1, 0] = ", diff(F2, VM1))
print("J[1, 1] = ", diff(F2, P2))
print("J[1, 2] = ", diff(F2, VM2))
print("")

print("COST FUNCTION w.r.t. X")
print("grad[0] = ", diff(cost, VM3))
print("grad[1] = ", diff(cost, VA3))
print("grad[2] = ", diff(cost, VA2))

print("COST FUNCTION w.r.t. U")
print("grad[0] = ", diff(cost, VM1))
print("grad[1] = ", diff(cost, P2))
print("grad[2] = ", diff(cost, VM2))