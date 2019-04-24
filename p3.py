# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 00:46:49 2019

@author: GiaHanXinhDep
"""


import numpy as np
import matplotlib.pyplot as plt
# Boost perfromance with jit. User can get rid of it.
from numba import jit
from scipy.sparse import spdiags 

@jit
def laplacian(N):
    """Construct a sparse matrix that applies the 5-point discretization"""
    e=np.ones(N**2)
    e2=([1]*(N-1)+[0])*N
    e3=([0]+[1]*(N-1))*N
    A=spdiags([-4*e,e2,e3,e,e],[0,-1,1,-N,N],N**2,N**2)
    return A

@jit
def main(M,N, Du, Dv, F, K):
    L = M
    u = np.ones((L, L))
    v = np.zeros((L, L))
    Lap = laplacian(L)

    h = L//2
    u += 0.02*np.random.random((L,L))
    v += 0.02*np.random.random((L,L))
    u[h-16:h+16, h-16:h+16] = 0.5
    v[h-16:h+16, h-16:h+16] = 0.25
    
    lu = u.reshape((L*L))
    lv = v.reshape((L*L))

    #evolve in time using Euler method
    for i in range(N):
        uvv = lu*lv*lv
        lu += (Du*Lap.dot(lu) - uvv +  F *(1-lu))
        lv += (Dv*Lap.dot(lv) + uvv - (F+K)*lv  )

        if i % 1000 == 0:
            filename = "./data/gs_{:02d}.png".format(i//1000)
            print(filename)
            plt.imshow(lu.reshape((L,L)), interpolation='bicubic',cmap=plt.cm.jet)
            plt.savefig(filename)

@jit
def bonus(M,N, Du, Dv, F, K):
    L = M
    u = np.ones((L, L))
    v = np.zeros((L, L))
    Lap = laplacian(L)

    h = L//2
    u += 0.02*np.random.random((L,L))
    v += 0.02*np.random.random((L,L))
    u[h-16:h+16, h-16:h+16] = 0.5
    v[h-16:h+16, h-16:h+16] = 0.25
    
    lu = u.reshape((L*L))
    lv = v.reshape((L*L))

    #evolve in time using Euler method
    for i in range(N):
        uvv = lu*lv*lv
        lu += (Du*Lap.dot(lu) - uvv +  F *(1-lu))
        lv += (Dv*Lap.dot(lv) + uvv - (F+K)*lv  )

    u = lu
    v = lv
    f = plt.figure(figsize=(25, 10), dpi=400, facecolor='w', edgecolor='k');
    sp =  f.add_subplot(1, 2, 1 );
    plt.pcolor(u.reshape((L, L)), cmap=plt.cm.jet)
    plt.axis('tight')

    sp =  f.add_subplot(1, 2, 2 );
    plt.pcolor(v.reshape((L, L)), cmap=plt.cm.jet)
    plt.axis('tight')
    plt.savefig("bonus.png")

if __name__ == "__main__":
#    Constances
    M=256
    N=32000
    Du=0.14
    Dv=0.06
    F=0.035
    k=0.065
    
#    0.14, 0.06, 0.035, 0.065
#    0.16, 0.08, 0.060, 0.062 
#    0.12, 0.08, 0.020, 0.050
#    0.16, 0.08, 0.035, 0.060
    
    
#    Calculate Du, Dx (No need)
    
#    Uncomment to gennerate images.
    main(M,N,Du,Dv,F,k)
    
    # bonus(M,N,Du,Dv,F,k)