import numpy as np

W_fh = 0
W_fx = 0
W_ih = 0
W_ix = 100
W_oh = 0
W_ox = 100
W_ch = -100
W_cx = 50


b_f = -100
b_i = 100
b_o = 0
b_c = 0


x = [1,1,0,1,1]

I = {}
F = {}
O = {}
C = {-1: 0}
H = {-1: 0}


def sigmoid(z):
    if z>=1:
        return 1
    elif z<=-1:
        return 0
    return (1/(1+np.exp(-z)))
    
def tanh(z):
    if z>=1:
        return 1
    elif z<=-1:
        return -1
    return np.tanh(z)


def f(t):
    if t in F:
        return F[t]

    F[t] = sigmoid(W_fh*h(t-1)+W_fx*x[t]+b_f)
    return F[t]


def i(t):
    if t in I:
        return I[t]
    I[t] = sigmoid(W_ih*h(t-1)+W_ix*x[t]+b_i)
    return I[t]


def o(t):
    if t in O:
        return O[t]

    O[t] = sigmoid(W_oh*h(t-1)+W_ox*x[t]+b_o)
    return O[t]


def c(t):
    if t in C:
        return C[t]
    C[t] = np.multiply(f(t), c(t-1))+np.multiply(i(t),
                                                 tanh(W_ch*h(t-1)+W_cx*x[t]+b_c))
    return C[t]


def h(t):
    if t in H:
        return H[t]
    H[t] = round(np.multiply(o(t), tanh(c(t))))
    return H[t]

h(4)
print(H)