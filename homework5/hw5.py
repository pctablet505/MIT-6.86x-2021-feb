# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:46:48 2021

@author: rahul
"""
# %%
# q1
from collections import defaultdict

e1 = (.6 * 10000 + .4 * 70000)
e2 = 37000
print(e1, e2)

# %%
# q2
e1 = (.6 * (10000 ** 2) + .4 * (70000 ** 2))
e2 = 37000 ** 2
print(e1, e2)

# %%
# q5b
a, b, c, d = 'abcd'
up = 'U'
down = 'D'
states = [a, b, c, d]

actions = [up, down]

R = {('a', 'U'): 1,
     ('b', 'U'): 1,
     ('b', 'D'): 1,
     ('c', 'U'): 10,
     ('c', 'D'): 1,
     ('d', 'D'): 10}
T = {('a', 'U'): b,
     ('b', 'U'): c,
     ('b', 'D'): a,
     ('c', 'U'): d,
     ('c', 'D'): b,
     ('d', 'D'): c}

V = defaultdict(int)
Q = defaultdict(int)
gamma = .75
# %%
new_V = V.copy()
for s in states:
    m = float('-inf')
    for act in actions:
        if (s == a and act == down) or (s == d and act == up):
            continue

        s1 = T[s, act]
        r = R[s, act] + gamma * V[s1]
        m = max(m, r)
    new_V[s] = m
V = new_V

print(V)
# %%
states = [0, 1, 2, 3, 4, 5]
C, M = 'CM'
actions = {C, M}
T = {(1, 'M', 0): 1,
     (1, 'C', 3): 0.7,
     (1, 'C', 1): 0.3,
     (2, 'M', 1): 1,
     (2, 'C', 4): 0.7,
     (2, 'C', 2): 0.3,
     (3, 'M', 2): 1,
     (3, 'C', 5): 0.7,
     (3, 'C', 3): .3,
     (4, 'M', 3): 1,
     (4, 'C', 4): 1,
     (5, 'M', 4): 1,
     (5, 'C', 5): 1}
R = defaultdict(int)

for s in states:
    for a in actions:
        if s != 0:
            R[s, a, s] = (s + 4) ** (-1 / 2)
        for s1 in states:
            if s != s1:
                R[s, a, s1] = abs(s1 - s) ** (1 / 3)

R[0, M, 0] = R[0, C, 0] = 0
s1s = {(1, 'M'): [0],
       (1, 'C'): [3, 1],
       (2, 'M'): [1],
       (2, 'C'): [4, 2],
       (3, 'M'): [2],
       (3, 'C'): [5, 3],
       (4, 'M'): [3],
       (4, 'C'): [4],
       (5, 'M'): [4],
       (5, 'C'): [5]}

Q = defaultdict(int)
V = defaultdict(int)

gamma = 0.6


def qviter(n=10):
    global Q, R, V

    for _ in range(n):
        new_Q = Q.copy()
        new_V = V.copy()
        for s in states:
            for a in actions:
                r = 0
                if (s, a) in s1s:
                    for s1 in s1s[s, a]:
                        r += T[s, a, s1] * (R[s, a, s1] + gamma * V[s1])
                new_Q[s, a] = r
            m = float('-inf')

            for a in actions:
                m = max(m, new_Q[s, a])

            new_V[s] = m

        V = new_V
        Q = new_Q

        for s in V:
            print(s, V[s])
        print()


qviter(1)
# %%
