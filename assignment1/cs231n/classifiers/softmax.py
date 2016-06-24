import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in xrange(num_train):
    score = X[i].dot(W)
   
    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    log_c = np.max(score)
    score -= log_c
    
    correct_score = score[y[i]]
    
    sum_i = np.sum(np.exp(score))
    p = np.exp(correct_score) / sum_i
    loss += - np.log(p)
       
    # Compute gradient
    # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
    # Here we are computing the contribution to the inner sum for a given i.
    for j in xrange(num_classes):
      pp = np.exp(score[j])/sum_i
      dW[:,j] += (pp-(j == y[i])) * X[i,:]
    
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
    
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train = X.shape[0]
  num_classes = W.shape[1]  

  score = X.dot(W)

  # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
  score -= np.max(score)

  correct_score = score[np.arange(num_train),y]  
  
  sum_i = np.sum(np.exp(score),axis=1)
    
  p = np.exp(correct_score) / sum_i  
      
  loss = np.sum(- np.log(p))
  loss /= num_train
 
    
  pp = np.exp(score)/  np.tile(sum_i, (num_classes, 1)).T
  
  ind = np.zeros(pp.shape)
  ind[range(num_train),y] = 1  

  dW= X.T.dot(pp-ind)
     
    
  dW /= num_train  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W  

  return loss, dW

