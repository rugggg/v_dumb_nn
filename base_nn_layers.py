import numpy as np
import pandas as pd
import math
from abc import ABC, abstractmethod
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles
# implementing some base NN modules from scratch

class layer(ABC):

    def __init__(self):
        pass
     
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self):
        pass

class net():

    def __init__(self, name: str):
        self.name = name
        self.layers = []


class mult(layer):
    def __init__(self):
        self.dx = None
        self.dy = None
        self.x = None
        self.y = None

    def forward(self, x: np.array, y: np.array) -> np.array:
        self.x = x
        self.y = y
        z = x*y
        # if we have a gradient 
        return z
    
    def backward(self, dz: np.array) -> list[np.array, np.array]:
        dx = self.y * dz
        dy = self.x * dz
        return [dx, dy]


class fcl(layer):
    """fully connected layer - keeping activation func seperate for now"""
    def __init__(self, width: int, output_width: int, activation="relu", name=None):
        self.width = width
        self.output_width = output_width
        self.W = np.random.randn(self.output_width, self.width) - 0.5
        # self.W = np.random.standard_normal((self.output_width, self.width)) * np.sqrt(2/self.width)
        self.b = np.zeros((self.output_width,1))
        self.activation=activation
        self.name = name
        print(self.name, self.W.shape)

        self.x = None # hold the input to this layer
        self.activation_cache = None # hold the activation output
        self.linear_cache = None # hold the linear layer output before activation
        self.dW = None # derivative wrt weights
        self.db = None # derivative wrt bias
        self.dz = None # derivative from upstream layers

    def forward(self, x: np.array):
        self.x = x # hold the inputs to this layer for backprop
        self.linear_cache = np.dot(self.W, x) + self.b # z is the linear activation output, z == linear cache
        # self.linear_cache = self.W @ x + self.b
        # then do the non-linearity activation function of choice on that linear activation output
        if self.activation == "relu":
            self.activation_cache = np.maximum(self.linear_cache, 0)
        if self.activation == "softmax":
            self.activation_cache = softmax(self.linear_cache)
        return self.activation_cache

    def backward(self, grad: np.array):
        # in a backward pass, we need to get the gradients with respect to x, W, and b

        # get the gradient of the 
        if self.activation == "relu":
            grad[self.activation_cache <= 0] = 0 # maybe need copy here?
            self.dz = grad
        if self.activation == "softmax":
            self.dz = grad
            # self.dz = np.diag(self.activation_cache) - np.dot(self.activation_cache, self.activation_cache.T)
        # linear backward
        m = self.x.shape[1]
        # print("dW calc inputs", self.name, self.dz.shape, self.linear_cache.shape)
        # print("self w shape:", self.W.shape)
        self.dW = 1/m * np.dot(self.dz, self.x.T) # is this right? self.x.T?
        self.db = 1/m * np.sum(self.dz, axis=1, keepdims=True)
        self.dA_prev = np.dot(self.W.T, self.dz)
        self.update()
        return self.dA_prev

    def update(self, lr=0.01):
        # so class has it's own grads already
        # print("updating with grad", self.dW, self.W[0])
        # print("updating:", self.name, self.width, self.output_width)
        # print(self.W.shape, self.dW.shape)
        # print(self.b.shape, self.db.shape)
        assert self.W.shape == self.dW.shape
        assert self.b.shape == self.db.shape
        self.W = self.W - self.dW*lr
        self.b = self.b - self.db*lr
        # print("set with grad", self.W[0])

    

# def ce_loss(y_pred, y_target):
#     return -np.sum(y_target * np.log(y_pred))/ y_target.shape[0]

def softmax(x):
    """Compute the softmax of vector x. on input that is shape (feat, batch), hence axis=0"""
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

def cross_entropy(y_true, y_pred):
    """
    Compute the cross-entropy of the predicted probability distribution y_pred and the true distribution y_true.
    on input that is shape (feat, batch), hence axis=0
    """
    assert y_true.shape == y_pred.shape
    return -np.sum(y_true * np.log(y_pred), axis=0)

def cross_entropy_grad(y_true, y_pred):
    """Compute the gradient of the cross-entropy loss function."""
    return -y_true / y_pred


class Network():

    def __init__(self):
        self.layers = [
            fcl(784, 64, "relu", "relu layer"),
            fcl(64, 32, "relu", "relu64layer"),
            fcl(32, 10, "softmax", "softmax layer"),
        ]

    def forward(self, x, y):
        curr_input = x
        for idx, l in enumerate(self.layers):
            curr_input = self.layers[idx].forward(curr_input)
        l = cross_entropy(y, curr_input)
        # print(curr_input.shape, y.shape)
        l_grad = cross_entropy_grad(y, curr_input) 
        return curr_input, l, l_grad

    def backward(self, loss_grad):
        # print("----backprop----")
        grad = loss_grad
        for l in reversed(self.layers):
            # print("grad:", grad.shape)
            if l.activation == "softmax":
                grad = l.backward(grad)
            else:
                grad = l.backward(grad)


def load_data(train_file, test_file=None):
    raw = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
    # print (raw.shape)

    #raw_test = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
    #print (raw_test.shape)

    Y = raw[:,0]     # first column of raw data are image labels
    X = raw[:,1:785]   # rest of the pixel data is in columns 1 to 785
    Y = Y.reshape(raw.shape[0],1)

    train_x = (X.T)/255.
    train_y = Y.T
    # targets = labels.to_numpy.reshape(-1)
    y_train_hot = np.squeeze(np.eye(10)[train_y])
    return train_x, y_train_hot.T

if __name__ == "__main__":
    

    X, Y = load_data("data/train.csv")
    print("SHAPES:", X.shape, Y.shape)
    nn = Network()
    losses = []

    for it in range(2000):
        #for sample_idx in range(300):
        #y_pred, loss = nn.forward(x[sample_idx], y[sample_idx])
        y_pred, loss, loss_grad = nn.forward(X, Y)
        #for yy in range(y_pred.shape[1]):
        #    assert y_pred[:, yy].sum() == 1
        # print("sum sanity check:", y_pred.sum(axis=0))
        # print("loss:", loss, loss_grad.shape)
        print("loss:", loss.mean())
        losses.append(loss.mean())
        if math.isnan(loss.mean()):
            print("NAN LOSS! Failed")
            break
        # print(loss_grad, loss_grad.shape)
        # print(loss_grad.sum(axis=1).shape)
        nn.backward(loss_grad)
        