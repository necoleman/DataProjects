# this is a quick python script to approximate the center of mass of an n-dim
# laminar plate sitting inside R^{n+1}, defined by the equations 
# \sum p_i = 1, p_j \leq c_j for all j, given constants c_j

import numpy as np
import matplotlib.pyplot as plt

# this recursively generates a fine mesh on a standard simplex of arbitrary dimension
# n: dimension (this is one less than the dimension of the space it lives in)
# mag: magnitude of the simplex (e.g. x+y+z = 0.4, x,y,z>0 is a 2-simplex with mag 0.4)
# dx: fineness of the mesh
def GenMesh(n,mag,dx):
    if n == 1:
        ret = []
        for j in range(int(mag/dx)):
            ret.append( [j*dx, mag-j*dx] )
        return ret
    else:
        ret = []
        for j in range(int(mag/dx)):
            m = mag - j*dx
            row = GenMesh(n-1,m,dx)
            for x in row:
                ret.append( [mag - m]+x )
        return ret

# inputs: list of constants c_j
def Center( bound_list ):
    bound_list = np.array(bound_list)
    n = bound_list.shape[0]
    dx = 0.005
    point_list = GenMesh(n-1,1,dx)
    for p in point_list:
        if np.any(np.array(p) > np.array(bound_list)):
            point_list.remove(p)
    a = np.array(point_list)
    return np.average(a,axis=0)
    

# JUST PUT YOUR NUMBERS HERE!
if __name__ == '__main__':
    print Center( [1.,1.0,1.0] )
