import numpy as np


def backward_prop(X, W1, b1, a1, W2, b2, y, y_hat):
    """
    Implements back propogation where the
    weights of the network are optimized using
    gradient descent

    Parameters:

    """
    nb1 = np.zeros(b1.shape)
    nb2 = np.zeros(b2.shape)
    nW1 = np.zeros(W1.shape)
    nW2 = np.zeros(W2.shape)

    # Gradient Descent
    nb2 = y - y_hat
    nW2 = nb2.dot(a1.T)
    nb1 = W2.T.dot(nb2)
    nW1 = nb1.dot(X.T)

    return nW1, nb1, nW2, nb2
