import numpy as np
from d2l import mxnet as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_params(vocab_size, num_hiddens, device):
  num_inputs = num_outputs = vocab_size
  
  def normal(shape):
    return np.random.normal(scale = 0.01, size = shape, ctx = device)
  
  def three():
    return (normal(num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)), np.zeros(num_hiddens, ctx = device))
    
  # update gate parameters
  w_xz, w_hz, b_z = three()
  # reset gate parameters
  w_xr, w_hr, b_r = three()
  # candidate hidden state parameters
  w_xh, w_hh, b_h = three()
  
  # output later parameters
  w_hq = normal((num_hiddens, num_outputs))
  b_q = np.zeros(num_outputs, ctx = device)
  
  params = [w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q]
  
  for param in params:
    param.attach_grad()
   
  return params

def gru(inputs, state, params):
  w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q = params
  H, = state
  outputs = []
  for X in inputs:
    Z = sigmoid(np.dot(X, w_xz) + np.dot(H, w_hz) + b_z)
    R = sigmoid(np.dot(X, w_xr) + np.dot(H, w_hr) + b_r)
    H_tilda = np.tanh(np.dot(X, w_xh) + np.dot(R * H, w_hh) + b_h)
    Y = np.dot(H, w_hq) + b_q
    outputs.append(y)
  return np.concatenate(outputs, axis = 0), (H,)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_later, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
        
