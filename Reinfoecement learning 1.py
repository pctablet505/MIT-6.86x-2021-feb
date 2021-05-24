# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 23:45:32 2021

@author: rahul
"""
from collections import defaultdict

S = [x for x in range(5)]
print(S)
n = len(S)

actions = {'left', 'right', 'stay'}
T = defaultdict(dict)
R = defaultdict(int)
for a in actions:
    R[4, a, -1] = 1


for s in S:

    for a in actions:
        if a == 'stay':
            if s == 0:
                T[s, a][s] = 1/2
                T[s, a][s+1] = 1/2
            elif s == n-1:
                T[s, a][s] = 1/2
                T[s, a][s-1] = 1/2
            else:
                T[s, a][s] = 1/2
                T[s, a][s-1] = 1/4
                T[s, a][s+1] = 1/4
        elif a == 'left':
            if s == 0:
                T[s, a][s] = 1/2
                T[s, a][s+1] = 1/2

            else:
                T[s, a][s] = 2/3
                T[s, a][s-1] = 1/3
        elif a == 'right':
            if s == n-1:
                T[s, a][s] = 1/2
                T[s, a][s-1] = 1/2

            else:
                T[s, a][s] = 2/3
                T[s, a][s+1] = 1/3

V = [0 for _ in range(n)]

    
    #R[s,a,3]
print(V)
eps = 1e-14
gamma = 0.5


def valueIteration(no_iter=10):
    for i in range(no_iter):
        for s in S:
            v_max = float('-inf')

            for a in actions:
                v = 0
                for s1 in T[s, a]:
                    if s==4:
                        v += T[s, a][s1]*(1+gamma*V[s1])
                    else:
                        v += T[s, a][s1]*(0+gamma*V[s1])
                v_max = max(v_max, v)
            V[s] = v_max
            
        print(i)
        #print([round(x, 3) for x in V])
        print(V)


valueIteration(200
               
               )
