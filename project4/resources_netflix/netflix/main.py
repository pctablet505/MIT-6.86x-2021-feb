from multiprocessing import Process
import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
'''

# TODO: Your code here
# %%
print('using k-means')
for K in range(1, 5):
    min_cost = float('inf')
    for seed in range(4):

        mixture, post = common.init(X, K, seed=seed)
        # common.plot(X, mixture, post,
        #             'num_clusters: {}'.format(K))
        mixture, post, cost = kmeans.run(X, mixture, post)
        # common.plot(X, mixture, post,
        #             'num_clusters: {}, cost: {}'.format(K, cost))

        min_cost = min(min_cost, cost)
    print(min_cost, K)


# %%
print('\n'*3)
print('now using em')

for K in range(1, 5):
    min_cost = float('inf')
    for seed in range(4):

        mixture, post = common.init(X, K, seed=seed)
        # common.plot(X, mixture, post,
        #             'num_clusters: {}'.format(K))
        mixture, post, cost = em.run(X, mixture, post)
        # common.plot(X, mixture, post,
        #             'num_clusters: {}, cost: {}'.format(K, cost))

        min_cost = min(min_cost, cost)
    print(min_cost, K)

# %%

# %%
K = 4
mixture, post = common.init(X, K)
mk, pk, ck = kmeans.run(X, mixture, post)
me, pe, ce = naive_em.run(X, mixture, post)

common.plot(X, mk, pk, 'kmeans, k: {} ,cost: {}'.format(K, ck))
common.plot(X, me, pe, 'naive_em, k: {}, cost: {}'.format(K, ce))

# %%
k = None
best_bic = float('-inf')
for K in range(1, 5):
    mixture, post = common.init(X, K)
    mixture, post, cost = naive_em.run(X, mixture, post)

    bic = common.bic(X, mixture, cost)
    if bic > best_bic:
        best_bic = bic
        k = K
print(k, best_bic)
'''
# %%


# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")
# print('''using test data''')

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")
print('''using netflix data''')

K = 12
for seed in range(5):
    mixture=post=cost=X_pred=None
    
    
    mixture, post = common.init(X, K,seed=seed)
    
    mixture,post, cost = em.run(X, mixture,post)
    
    print(K, seed, cost)
    X_pred = em.fill_matrix(X, mixture)
    rmse = common.rmse(X_pred, X_gold)
    print(rmse)

