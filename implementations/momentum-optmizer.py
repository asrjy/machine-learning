# Note: Forward and Backward Propogation functions used here are problem specific
def sigmoid(z):
  '''In this function, we will compute the sigmoid(z)'''
  # we can use this function in forward and backward propagation
  return 1/(1 + np.exp(-z))

def forward_propagation(X, y, W):
  exp = np.exp(np.dot(W[0]*X[0] + W[1]*X[1], W[0]*X[0] + W[1]*X[1]) + W[5])  
  tanh = np.tanh(exp + W[6]) 
  sig_grader = (sigmoid(np.sin(W[2]*X[2])*(W[3]*X[3]+W[4]*X[4]) + W[7]))
  sig = sig_grader * W[8]
  y_dash = tanh + sig
  l = (y - y_dash)**2
  data = {
    "dy_pr" : -2 * (y - y_dash),
    "loss" : (y - y_dash)**2,
    "exp" : exp,
    "tanh" : tanh,
    "sig" : sig,
    "sigmoid" : sig_grader,
    "y" : y,
    "y_dash" : y_dash,
    "dw1" : -2 * (y - y_dash) * (2 * (1 - tanh**2) * exp * np.dot((W[0]*X[0])+(W[1]*X[1]), X[0])), 
    "dw2" : -2 * (y - y_dash) * (2 * (1 - tanh**2) * exp * np.dot((W[0]*X[0])+(W[1]*X[1]), X[1]))
  }
  return data


def backward_propagation(X,W,dic):
  common = -2 * (dic["y"] - dic["y_dash"])
  tanh = dic["tanh"]
  exp = dic["exp"]
  sig = dic["sigmoid"]
  d_sig = sig * (1 - sig)
  d_tanh = (1 - tanh**2)
  data = {
      "dw1": dic["dw1"],
      "dw2": dic["dw2"],
      "dw3": common * W[8] * d_sig * np.dot((W[3]*X[3] + W[4]*X[4]), np.cos(W[2] * X[2])) * X[2],
      "dw4": common * W[8] * d_sig * np.dot(np.sin(W[2]*X[2]), X[3]),
      "dw5": common * W[8] * d_sig * np.sin(W[2] * X[2]) * X[4] ,
      "dw6": common * d_tanh * exp,
      "dw7": common * d_tanh,
      "dw8": common * W[8] * d_sig,
      "dw9": common * sig 
  }
  return data


W = np.random.normal(0, 0.01, 9)
momentum_updates = []
eta = 0.01
gamma = 0.9
batch_size = 200
first_it = 1
for epoch in tqdm(range(100)):
  indices = np.random.randint(low = 0, high = 505, size = batch_size)
  for index in indices:
    fp = forward_propagation(X[index], y[index], W)
    bp = backward_propagation(X[index], W, fp)
    dW = [bp["dw1"], bp["dw2"], bp["dw3"], bp["dw4"], bp["dw5"], bp["dw6"], bp["dw7"], bp["dw8"], bp["dw9"]]
    if first_it == 1:
      v = eta * np.array(dW)
      W -= v
      prev_v = v
    else:
      v = prev_v - (eta * np.array(dW))
      W -= v
      prev_v = v
  momentum_updates.append(fp["loss"])
