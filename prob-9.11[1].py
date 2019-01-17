#script: prob-9.11.py
#tests Gauss elimination algorithm
#author: Rahul Shridhar

import numpy as np
import _linalg

x1 = np.array( [2, -3, -8] )  #Coefficients of x1
x2 = np.array( [-6, -1, 1] )  #Coefficients of x2
x3 = np.array( [-1, 7, -2] )  #Coefficients of x3
xvars = ["x1", "x2", "x3"]

A = np.array( [[x1[0], x2[0], x3[0]],
               [x1[1], x2[1], x3[1]],
               [x1[2], x2[2], x3[2]]] )
b = np.array( [-38, -34, -20] )
#b = m*g - c*v  #numpy arrays can operate on scalars!
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
