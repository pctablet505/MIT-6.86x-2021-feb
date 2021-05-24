# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:39:35 2021

@author: rahul
"""

from collections import Counter
X = '''A B A B B C A B A A B C A C'''.split()
C = Counter(X)
P = {x: C[x]/len(X) for x in C}


def prob(seq):
    p = 1
    for x in seq:
        p *= P[x]
    print(p)


prob('ABC')
prob('BBB')
prob('ABB')
prob('AAC')
print('\n'*3)


s = sorted(set(X))

X='ABABBCABAABCAC'
def count(x):
    c = 0
    for i in range(len(X)-1):
        if x == X[i:i+2]:
            c += 1
    return c


C = {}
for a in s:
    for b in s:
        x = a+b
        C[x]=count(x)
s=sum(C.values())
P = {x: C[x]/s for x in C}
def prob(seq):
    p = 1/3
    for i in range(len(seq)-1):
        p*=P[seq[i:i+2]]
    
    print(p)
prob('AABCBAB')
