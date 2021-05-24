# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:56:32 2021

@author: rahul
"""
import matplotlib.pyplot as plt

import numpy as np


def l1(a, b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


def l2(a, b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5


X = [[0, -6], [4, 4], [0, 0], [-5, 2]]

k = 2
n = len(X)
Z = [[-5, 2], [0, -6]]


C = {}
present_cost = 0

xp = [x for x, y in X]
yp = [y for x, y in X]
#%%


def kmedl1():
    global C, X, Z, n, k, present_cost
    while True:

        cost = 0
        for i in range(n):
            m = float('inf')
            for j in range(k):
                d = l1(X[i], Z[j])
                if d < m:
                    C[i] = j
                    m = d
            cost += d
        present_cost = cost

        for j in range(k):
            m = float('inf')
            for i1 in range(n):
                if C[i1] == j:
                    s = 0
                    for i2 in range(n):
                        if C[i2] == j:
                            s += l1(X[i1], X[i2])

                    if s < m:
                        m = s
                        Z[j] = X[i1]
        print(present_cost)
        print(C)
        print(Z)

        c = [C[i] for i in range(n)]
        plt.figure(figsize=(4, 4))
        plt.grid()
        plt.scatter(xp, yp, c=c)
        plt.scatter(np.array(Z)[:, 0], np.array(Z)[:, 1], marker='+')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.show()
        if input() == 'n':
            break


#%%
kmedl1()
print(present_cost)
print(C)
print(Z)
#%%
def kmedl2():
    global C,X,Z,n,k,present_cost
    while True:
        
        cost=0
        for i in range(n):
            m=float('inf')
            for j in range(k):
                d=l2(X[i],Z[j])
                if d<m:
                    C[i]=j
                    m=d
            cost+=d
        present_cost=cost
        
        for j in range(k):
            m=float('inf')
            for i1 in range(n):
                if C[i1]==j:
                    s=0
                    for i2 in range(n):
                        if C[i2]==j:
                            s+=l2(X[i1],X[i2])
                    
                    if s<m:
                        m=s
                        Z[j]=X[i1]
        print(present_cost)
        print(C)
        print(Z)
        
        c=[C[i] for i in range(n)]
        plt.figure(figsize=(4,4))
        plt.grid()
        plt.scatter(xp,yp,c=c)
        plt.scatter(np.array(Z)[:,0],np.array(Z)[:,1],marker='+')
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.show()
        if input()=='n':
            break

#%%
kmedl2()
print(present_cost)
print(C)
print(Z)    

#%%

def kmeansl1():
    global C, X, Z, n, k, present_cost
    while True:

        cost = 0
        for i in range(n):
            m = float('inf')
            for j in range(k):
                d = l1(X[i], Z[j])
                if d < m:
                    C[i] = j
                    m = d
            cost += d
        present_cost = cost
        
        
        for j in range(k):
            s0=s1=0
            count=0
            for i1 in range(n):
                if C[i1] == j:
                    count+=1
                    s0+=X[i1][0]
                    s1+=X[i1][1]
            r0,r1=None,None
            s0/=count
            s1/=count
            m=float('inf')
            for i in range(n):
                if C[i]==j:
                    d=l1(X[i],[s0,s1])
                    if d<m:
                        m=d
                        r0,r1=X[i]
            Z[j]=[r0,r1]
                
        print(present_cost)
        print(C)
        print(Z)

        c = [C[i] for i in range(n)]
        plt.figure(figsize=(4, 4))
        plt.grid()
        plt.scatter(xp, yp, c=c)
        plt.scatter(np.array(Z)[:, 0], np.array(Z)[:, 1], marker='+')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.show()
        if input() == 'n':
            break
#%%
kmeansl1()
print(present_cost)
print(C)
print(Z)    