# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 01:31:52 2021

@author: rahul
"""

import numpy as np

alpha = 0.75
gamma = 0.5

s1, s2 = 's1', 's2'
a1, a2 = 'a1', 'a2'
actions = {a1, a2}
states = {s1, s2}

R = {}
T = {}
Q = {}

for s in states:
    for a in actions:

        Q[s, a] = 0


def update(s, a, s1, r):

    m = float('-inf')
    for a1 in actions:
        m = max(m, Q[s1, a1])

    Q[s, a] = (alpha*r+gamma*m)+(1-alpha)*Q[s, a]
    print(Q)


samples = [(s1, a1, s1, 1),
           (s1, a1, s2, -1),
           (s2, a2, s1, 1), ]

for s,a,s1,r in samples:
    update(s,a,s1,r)