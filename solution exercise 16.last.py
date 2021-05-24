# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
from scipy.stats import norm
from collections import defaultdict
sqrt = np.sqrt


def prob(mu, sig_2, x):
    return (1/sqrt((2*np.pi*sig_2)))*np.exp(-((x-mu)**2)/(2*sig_2))


muj = [-3, 2]
pj = [.5, .5]
x = [.2, -.9, -1, 1.2, 1.8]
sigmasq = [4, 4]

N = []
pji = defaultdict(list)

for j in range(2):
    temp_p = []
    for i in range(len(x)):
        p = pj[0]*prob(muj[j], sigmasq[j], x[i])
        temp_p.append(p)
    pji[j] = temp_p

for j in range(2):
    for i in range(len(x)):
        s_i=pji[0][i]+pji[1][i]
        pji[0][i]/=s_i
        pji[1][i]/=s_i

print("solution question 1")
for t in pji[0]:
    print(t)
print('\n'*2)

for j in range(2):
    s1=0
    
    pj[j]=(1/len(x))*sum(pji[j])
    
    for i in range(len(x)):
        s1+=pji[j][i]*x[i]
    muj[j]=s1/sum(pji[j])
    
    s2=0
    for i in range(len(x)):
        s2+=(pji[j][i]*abs(x[i]-muj[j])**2)
    s2/=(sum(pji[j]))*1
    sigmasq[j]=s2
    
print("solution question 2")
print(pj[0])
print(muj[0])
print(sigmasq[0])

