import numpy as np


def ReLU(R):
    return np.maximum(R, 0)


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def forward_propogation(X, W1, b1, W2, b2):
    """
    Implements forward propogation step

    Parameters:

    """
    Z1 = W1.dot(X) + b1
    R1 = ReLU(Z1)
    Z2 = W2.dot(R1) + b2
    return sigmoid(Z2)
