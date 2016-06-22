import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

 # Average gradients as well
  dW /= num_train

  # Add regularization to the gradient
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
 
  num_train = X.shape[0]  
  scores = X.dot(W)  
  correct_class_score = scores[np.arange(num_train), y]
    
  margin = (scores.T - correct_class_score).T + 1
  margin[np.arange(num_train), y] = 0
  
  loss = np.sum(margin[margin >0])  / num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)  
    
  num_pos = np.sum(margin > 0, axis=1) # number of positive losses

  dscores = np.zeros(scores.shape)

  dscores[margin > 0] = 1
  dscores[np.arange(num_train),y] = -num_pos

  # Average gradients as well
  dW = X.T.dot(dscores) / num_train 

  # Add regularization to the gradient
  dW += reg * W  

  return loss, dW
