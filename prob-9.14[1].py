#script: prob-9.14.py
#tests Gauss elimination algorithm
#author: Rahul Shridhar

import numpy as np
import _linalg

m = np.array( [55, 75, 60, 75, 90] )  #mass of parachutists
c = np.array( [10, 12, 15, 16, 10] )  #drag coefficients of parachutists
v = 9  #velocity of parachutists
g = 9.81  #gravity constant
xvars = ["a", "T12", "T23", "T34", "T45"]

A = np.array( [[m[0], 1, 0, 0, 0],
               [m[1], -1, 1, 0, 0],
               [m[2], 0, -1, 1, 0],
               [m[3], 0, 0, -1, 1],
               [m[4], 0, 0, 0, -1]] )
b = m*g - c*v  #numpy arrays can operate on scalars!
#equivalent to:
#b[0] = m[0]*g - c[0]*v
#b[1] = m[1]*g - c[1]*v
#b[2] = m[2]*g - c[2]*v

print("A =\n{}".format(A))
print("b = {}".format(b))

print("solving...")
x = _linalg.gauss(A, b)  #solve using Gauss elimination method w/ scaling & partial pivoting
x1 = np.linalg.solve(A, b)  #solve using numpy (to test correctness)
print("x = {}\n".format(x))
print("x1 = {}\n".format(x1))

if x is not None:
    for i in range(len(A)):
        print("{} = {}".format(xvars[i], x[i]))
