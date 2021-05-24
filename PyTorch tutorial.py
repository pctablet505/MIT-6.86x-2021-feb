# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:21:01 2021

@author: rahul
"""

# %%
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_classification
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import tqdm


# %%
# construct a bunch of ones
some_ones = torch.ones(2, 2)
print(some_ones)

# Construct a bunch of zeros
some_zeros = torch.zeros(2, 2)
print(some_zeros)

# Construct some normally distributed values
some_normals = torch.randn(2, 2)
print(some_normals)

# %%
torch_tensor = torch.randn(5, 5)
numpy_ndarray = torch_tensor.numpy()
back_to_torch = torch.from_numpy(numpy_ndarray)

# %%
# create two tensors
a = torch.randn(5, 5)
b = torch.randn(5, 5)
print(a)
print(b)

# %%
# indexing by i,j
another_tensor = a[2, 2]
print(another_tensor)

# %%
first_row = a[0, :]
first_column = a[:, 0]
combo = a[2:4, 2:4]
print(first_row)
print(first_column)
print(combo)

# %%
# addition
c = a+b
print(c)

# elementwise multiplication: c_ij = a_ij*b_ij
c = a*b
print(c)

# Matrix vector multiplication
c = a.mm(b)
print(c)

# matrix vector 5x5 * 5x5 --> 5
vec = a[:, 0]
vec_as_matrix = vec.view(5, 1)
v2 = a@vec_as_matrix
print(v2)

# %%

# inplace operations
# add 1 to all elements
a.add_(1)

# divide all elements by 2
a.div_(2)

# set all emements to 0
a.zero_()

# %%
"""Manipulate dimensions..."""
# add a dummy dimension, e.g. (n,m) --> (n,m,1)
a = torch.randn(10, 10)
# at the end
print(a.unsqueeze(-1).size())

# at the beginig
print(a.unsqueeze(0).size())

# in the middle
print(a.unsqueeze(1).size())

# what you give you can take away
print(a.unsqueeze(0).squeeze(0).size())

# view things differently, i.e. flat
print(a.view(100, 1).size())

# or not flat
print(a.view(50, 2).size())

# copyn data across a new dimension!
a = torch.randn(2)
a = a.unsqueeze(-1)
print(a)

print(a.expand(2, 3))

# %%
# check if you have a GPU
do_i_have_cuda = torch.cuda.is_available()
print(do_i_have_cuda)
print(a.cuda())


# %%
# batched matrix multiply
a = torch.randn(10, 5, 5)
b = torch.randn(10, 5, 5)

# the same as for i in 1 ... 10, c_i = a[i].mm(b[i])
c = a.bmm(b)
print(c.size())

# %%

# A tensor that will remember gradients
x = torch.randn(10, requires_grad=True)
print(x)

# %%
x_a = torch.randn(1, requires_grad=True)
x_b = torch.randn(1, requires_grad=True)
x = x_a*x_b
x1 = x**2
x2 = 1/x1
x3 = x2.exp()
x4 = 1+x3
x5 = x4.log()
x6 = x5**(1/3)
x6.backward()

# %%
# Manual Neural Net + Autograd SGD example


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
# %%


# get ourselves a simple dataset

set_seed(7)
X, Y = make_classification(n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1)
# %%
print('No of examples: {}'.format(X.shape[0]))
print('No of feature: {}'.format(X.shape[1]))

# take a peak
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolors='k')
plt.show()
# %%
# convert data to pytorch
X, Y = torch.from_numpy(X), torch.from_numpy(Y)
X, Y = X.float(), Y.float()
# %%
# define dimensions
num_features = 2
hidden_size = 100
num_outputs = 1

# learning rate
eta = 0.01
num_steps = 1000

# %%
# input to hidden weights
W1 = torch.randn(hidden_size, num_features, requires_grad=True)
b1 = torch.zeros(hidden_size, requires_grad=True)

# hidden to output
W2 = torch.randn(num_outputs, hidden_size, requires_grad=True)
b2 = torch.zeros(num_outputs, requires_grad=True)
# Group parameters
parameters = [W1, b1, W2, b2]

# Get random order
indices = torch.randperm(X.size(0))
avg_loss = []

for step in range(num_steps):
    # get example
    i = indices[step % indices.size(0)]
    x_i, y_i = X[i], Y[i]

    # run example
    hidden = torch.relu(W1@(x_i)+b1)
    y_hat = torch.sigmoid(W2@(hidden)+b2)
    eps = 1e-6
    loss = -(y_i*(y_hat+eps).log()+(1-y_i)*(1-y_hat+eps).log())

    # add to our running average learning curve. Don't forget .item()!
    if step == 0:
        avg_loss.append(loss.item())
    else:
        old_avg = avg_loss[-1]
        new_avg = (loss.item()+old_avg*len(avg_loss))/(len(avg_loss)+1)
        avg_loss.append(new_avg)

    # zero out all previous gradients
    for param in parameters:
        # it might start out as None
        if param.grad is not None:
            # inplace
            param.grad.zero_()
    # Backward pass
    loss.backward()

    # update parameters
    for param in parameters:
        # inplace
        param.data = param.data-eta*param.grad

plt.plot(range(num_steps), avg_loss)
plt.ylabel('Loss')
plt.xlabel('Step')
plt.show()

# %%
# torch.nn

linear = nn.Linear(10, 10)
print(linear)

conv = nn.Conv2d(1, 20, 5, 1)
print(conv)

rnn = nn.RNN(10, 10, 1)
print(rnn)

# %%
print(linear.weight)
print([k for k, v in conv.named_parameters()])
# %%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()

# %%
optimizer = optim.SGD(model.parameters(), lr=0.01)
# %%


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for data, target in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print('Train Epoch: {}\t Loss: {:.6f}'.format(
        epoch, total_loss/len(train_loader)))

# %%


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy :{}/{} ({:.-f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))

# %%


from torchvision import datasets, transforms

# See the torch DataLoader for more details.
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True)
# %%


for epoch in range(1, 10+1):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
