import tensorflow as tf
from d2l import tensorflow as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def get_lstm_params(vocab_size, num_hiddens):
  num_inputs = num_outputs = vocab_size
  
  def normal(shape):
    return tf.Variable(tf.random.normal(shape=shape, stddev = 0.01, mean = 0, dtype = tf.float32))
  
  def three():
    return normal((num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)), tf.Variable(tf.zeros(num_hiddens), dtype = tf.float32)
  
  # Input Gate Parameters
  w_xi, w_hi, b_i = three()
  
  # Forget Gate Parameters
  w_xf, w_hf, b_f = three()
  
  # Output Gate Paramteres
  w_xo, w_ho, b_o = three()
  
  # Candidate Memeory Cell Parameters
  w_xc, w_hc, b_c = three()
  
  # Output Layer Parameters
  w_hq = normal((num_hiddens, num_outputs))
  b_q = tf.Variable(tf.zeros(num_outputs), dtype = tf.float32)
  
  # Attach Gradients
  params = [w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hq, b_q]
  return params

def init_lstm_state(batch_size, num_hiddens):
  return (tf.zeros(shape = (batch_size, num_hiddens)), tf.zeros(shape = (batch_size, num_hiddens)))

def lstm(inputs, state, params):
  w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hq, b_q = params
  (H, C) = state
  outputs = []
  for X in inputs:
    X = tf.reshape(X, [-1, w_xi.shape[0])
    # Input Gate
    I = tf.sigmoid(tf.matmul(X, w_xi) + tf.matmul(H, w_hi) + b_i)
    # Forget Gate
    F = tf.sigmoid(tf.matmul(X, w_xf) + tf.matmul(H, w_hf) + b_f)
    # Output Gate
    O = tf.sigmoid(tf.matmul(X, w_x0) + tf.matmul(H, w_ho) + b_o)
    # Cell State
    C_tilda = tf.tanh(tf.matmul(X, w_xc) + tf.matmul(H, w_hc) + b_c)
    C = F * C + I * C_tilda
    H = O * tf.tanh(C)
    Y = tf.matmul(H, w_hq) + b_q
    outputs.append(Y)
  return  tf.concat(outputs, axis = 0), (H, C)

vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu().__device_name
num_epochs, lr = 500, 1
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
  model = d2l.RNNModelScratch(len(vocab), num_hiddens, init_lstm_state, lstm, get_lstm_params)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
