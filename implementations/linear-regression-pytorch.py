%matplotlib inline
import random
import torch

# Generating Dataset
def synthetic_data(w, b, num_examples):
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = torch.matmul(X, w) + b
  y += torch.normal(0, 0.01, y.shape)
  return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# Reading the Dataset
# Minibatch 
def data_iter(batch_size, features, labels):
  num_examples = len(features)
  indices = list(range(num_examples))
  # The examples are read at random, in no particular order
  random.shuffle(indices)
  for i in range(0, num_examples, batch_size):
    batch_indices = torch.tensor(
      indices[i: min(i + batch_size, num_examples)])
    yield features[batch_indices], labels[batch_indices]

# Initializing Model Parameters
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Defining Model
def linreg(X, w, b): 
  return torch.matmul(X, w) + b

# Defining the Loss Function
def squared_loss(y_hat, y): 
  return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# Defining the Optimization Algorithm
def sgd(params, lr, batch_size): #@save
  """Minibatch stochastic gradient descent."""
  with torch.no_grad():
  for param in params:
    param -= lr * param.grad / batch_size
    param.grad.zero_()

# Training 
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss


for epoch in range(num_epochs):
  for X, y in data_iter(batch_size, features, labels):
    l = loss(net(X, w, b), y) # Minibatch loss in `X` and `y`
    # Compute gradient on `l` with respect to [`w`, `b`]
    l.sum().backward()
    sgd([w, b], lr, batch_size) # Update parameters using their gradient
  with torch.no_grad():
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')


